# img2tag2img
A ComfyUI workflow and support tools for large scale indirect img2img using taggers and LLM. Images are "intepreted" using a tagger and local LLM, and these are used to build a prompt for the next round of generations. 

The cycle [can be repeated for as long as you have GPU cycles and disk to spare](https://en.wikipedia.org/wiki/I_Am_Sitting_in_a_Room).

These tools were recently rewritten to focus on configuration from [.toml files](https://toml.io/en/) instead of from a combination of an .ini file and command line switches. This is a breaking change - apologies - and also requires the installation of `tomlkit` because native toml support is only available in very recent python builds. Reading the example files should show you how to convert an old .ini file across, if you have one.

## example

Example context - the script is running in WSL, Comfy instance is running in Windows. The Comfy output folder is mounted in WSL in `/home/curious/sdoutput`. Input files are in a subfolder of Comfy's output folder called `stage1`. These files are read, tagged, and regenerated into a subfolder named `stage2`. Once this is done and the queue is clear, those files are in turn read and regenerated into `stage3`. Neither `stage2` or `stage3` exist yet.


```sh
(venv) curious@XXXX:~/usrmnt/gh/img2tag2img$ python post_to_api.py workflows/i2t2i.sdxl.raw.json stage1 stage2 stage3
Current configuration: {'latent': {'landscape': [1216, 832], 'portrait': [832, 1216], 'square': [1024, 1024]}, 'llava': {'disable': False, 'model': 'llama/llava-v1.5-7b-Q4_K', 'projector': 'llama/llava-v1.5-7b-mmproj-Q4_0', 'prompt': 'Please describe this image in detail.'}, 'lora': [{'name': 'pony/style/Smooth Anime 2 Style SDXL_LoRA_Pony Diffusion V6 XL.safetensors', 'strength': 0.8}, {'name': 'pony/style/Concept Art Twilight Style SDXL_LoRA_Pony Diffusion V6 XL.safetensors', 'strength': 0.8}, {'name': 'pony/style/Rainbow Style SDXL_LoRA_Pony Diffusion V6 XL.safetensors', 'strength': 0.8, 'trigger': 'colorful'}, {'name': 'None'}], 'misc': {'save_predetail': False}, 'model': {'checkpoint': ['sdxl/ponyFaetality_v10.safetensors', 'sdxl/ponyDiffusionV6XL_v6StartWithThisOne.safetensors'], 'force_lora': [{'name': 'None'}]}, 'overload': {'merge': {'load_model': {'inputs': {'batch_size': 1}}}}, 'paths': {'archive': '/home/curious/sdoutput/', 'output': ''}, 'prompt': {'preamble': 'score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, rating_safe', 'negative': 'head_out_of_frame, 3d'}, 'server': {'address': '127.0.0.1', 'on_windows': True, 'poll_delay': 0.25, 'port': 8188}, 'wd14': {'banned': ['watermark', 'web_address']}}
Workflow comment was: What you should provide to this workflow: load_model.base_ckpt_name; save_image.modelname; preamble.string, negative_prompt.string, full_path.string, wd14_tagger.exclude_tags, llava_tagger.model, llava_tagger.mm_proj, llava_tagger.prompt, seed.seed, image_loader.image_data
** standing by to run workflow from stage1 to stage2
   queue has emptied, time is 2024-05-17 12:37:55.934428
   enumerating images in /home/curious/sdoutput/stage1
   found 4 files
   .... - complete
** standing by to run workflow from stage2 to stage3
   queue has emptied, time is 2024-05-17 12:42:17.083481
   enumerating images in /home/curious/sdoutput/stage2
   found 4 files
   .... - complete
```

<p align="center">
<img src="https://github.com/curiousjp/img2tag2img/assets/48515264/ba286898-52b9-4faf-a320-6745730fc087"/>
</p>

## the default workflow

The default workflow is designed for use with SDXL, but one is also provided for sd15. It relies on the following custom node packs:

* [Efficiency Nodes](https://github.com/jags111/efficiency-nodes-comfyui)
* [Image Saver](https://github.com/alexopus/ComfyUI-Image-Saver)
* [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
* [Inspire Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack)
* [LLaVA Captioner](https://github.com/ceruleandeep/ComfyUI-LLaVA-Captioner)
* [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui)
* [WD14 Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)

The workflow is fairly simple and uses descriptive names for its nodes. If there's a dependency on here you'd rather avoid, you should be able to work out how to remove it fairly easily.

The exception is the LLaVA component, which can be hard to set up and some versions of the backend library seem to leak GPU memory - some notes towards a bandaid fix can be found [here](https://github.com/ceruleandeep/ComfyUI-LLaVA-Captioner/issues/11). For this reason, if it is simply too much trouble to be bothered with, you can remove it from the workflow at runtime by specifying the `disable` key as true in the `llava` section of your configuration file. The `default.sd15.toml` file shows this in action.

## how is the program configured

When the program starts, it reads a configuration file (by default, `default.toml`, although you can override this with `--config_file`), a workflow, and your selection of a source folder and then one or more output folders.

The program then loads the specified workflow file, and, for each image, begins to modify it with the values taken from the configuration file. If you specified the `--dump` command line option, the new workflows are written to disk, otherwise they are submitted over the API to Comfy.

Dumping the preprocessed workflows isn't really compatible with chaining multiple output folders together - as the workflows aren't submitted, there will be nothing in the sucessive folders to generate further workflows from.

## customising configuration

While I have tried to provide sensible defaults for most settings, a configuration file must at least provide the `checkpoint` value in the `model` section and the `archive` value in the `paths` section. `default.toml` contains every key and section recognised by the program, and is commented to explain what each one does. While lots of keys are available, a general override mechanism is also provided to allow you to merge or replace sections of your workflow at will.

## other notes

It is useful (for me) to be able to easily compare images in the same lineage. This script tries to promote this with how it names the output files. If the input file has a filename starting with four digits, or an X followed by four digits, those same digits will be used on the output of the process. If the file does _not_ have a name matching this pattern, the program will attempt to assign one (although this may have mixed results where some files have this prefix and some do not.) You can then use a program like [Diffusion.Toolkit](https://github.com/RupertAvery/DiffusionToolkit) to filter to a specific parent folder and then sort by file name to bring all the relevant items together.