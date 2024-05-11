# img2tag2img
A ComfyUI workflow and support tools for large scale indirect img2img using taggers and LLM. Images are "intepreted" using a tagger and local LLM, and these are used to build a prompt for the next round of generations. 

The cycle [can be repeated for as long as you have GPU cycles and disk to spare](https://en.wikipedia.org/wiki/I_Am_Sitting_in_a_Room).
## example

```sh
(venv) curious@XXXX:~/usrmnt/gh/img2tag2img$ python post_to_api.py workflows/i2t2i.sdxl.raw.json samples/stage1 samples/stage2 samples/stage3
Current configuration: {'archive_path': '/home/curious/sdoutput/', 'banned_tags': 'watermark, web_address', 'checkpoint': 'sdxl/ponyDiffusionV6XL_v6StartWithThisOne.safetensors', 'comfy_address': '127.0.0.1', 'comfy_on_windows': True, 'comfy_port': '8188', 'llava_model': 'llama/llava-v1.5-7b-Q4_K', 'llava_projector': 'llama/llava-v1.5-7b-mmproj-Q4_0', 'llava_prompt': 'Please describe this image in detail.', 'lora_prefix': 'pony/style/', 'lora_root': '/home/curious/lora/', 'negative': 'head_out_of_frame, 3d', 'output_path': '', 'preamble': 'score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, rating_safe'}
Workflow comment was: What you MUST provide to this workflow: load_model.base_ckpt_name; save_image.modelname; preamble.string, negative_prompt.string, full_path.string, wd14_tagger.exclude_tags, llava_tagger.model, llava_tagger.mm_proj, llava_tagger.prompt, seed.seed, image_loader.image_data
LoRA discovery: 39 LoRA detected (including 'blank' LoRA options).
** standing by to run workflow from samples/stage1 to samples/stage2
   queue has emptied, time is 2024-05-11 20:47:54.453331
   enumerating images in /home/curious/sdoutput/samples/stage1
   found 4 files
   .... - complete
** standing by to run workflow from samples/stage2 to samples/stage3
   queue has emptied, time is 2024-05-11 20:51:48.243689
   enumerating images in /home/curious/sdoutput/samples/stage2
   found 4 files
   .... - complete
```

<p align="center">
<img src="https://github.com/curiousjp/img2tag2img/assets/48515264/ba286898-52b9-4faf-a320-6745730fc087"/>
</p>

## the default workflow

The provided workflow is designed for use with SDXL. It cannot be loaded into ComfyUI by itself without further tweaking. It relies on the following custom node packs:

* [ASTERR](https://github.com/WASasquatch/ASTERR)
* [CrasH Utils](https://github.com/chrish-slingshot/CrasHUtils)
* [Efficiency Nodes](https://github.com/jags111/efficiency-nodes-comfyui)
* [Image Saver](https://github.com/alexopus/ComfyUI-Image-Saver)
* [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
* [Inspire Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack)
* [LLaVA Captioner](https://github.com/ceruleandeep/ComfyUI-LLaVA-Captioner)
* [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui)
* [WD14 Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger)

The workflow is fairly simple and uses descriptive names for its nodes. If there's a dependency on here you'd rather avoid, you should be able to work out how to remove it fairly easily.

The exception is the LLaVA component, which can be hard to set up and some versions of the backend library seem to leak GPU memory - some notes towards a bandaid fix can be found [here](https://github.com/ceruleandeep/ComfyUI-LLaVA-Captioner/issues/11). For this reason, if it is simply too much trouble to be bothered with, you can use the `--disable_llm` option.

## configuration

When the program starts, it reads a configuration file (by default, `default.ini`, although you can override this with `--config_file`), and then reads any command line switches you may have set. Command line switches will override anything given in the configuration file.

The program then loads the specified workflow file, and begins to modify it with the values taken from the configuration file and the switches. 

### mandatory configuration

Some configuration settings have default values, but others must be set, either in the file on the command line, for the program to run:

* `archive_path` when specifying the folders of images to read from and write to, they are given relative to this folder. So, if your images are in `C:\Comfy\Output\part1`, you might set your `archive_path` to `C:\Comfy\Output\`.
* `banned_tags` if wd14tagger keeps detecting watermarks or other undesirable false positives, you can list them here. They should be separated by commas.
* `checkpoint` is the path to your stable diffusion checkpoint, including the file extension. It should be given relative to your ComfyUI checkpoint folder.
* `llava_model`, `llava_projector` are the paths to these two models, relative to the `models` folder inside the ComfyUI-LLaVA-Captioner folder. You don't need to set these if you're using `--disable_llm`.
* `llava_prompt`, the prompt given to LLaVA when describing your image. Specific image domains can benefit from tweaking this, but the example given in the `default.ini` file will probably work for most people.
* `lora_root`, the path to your LoRA folder. If you don't want to apply LoRA to your images, you can point this at an empty folder.
* `negative`, a negative prompt.
* `preamble`, a fixed string (like `masterpiece, best quality` etc) to add to the front of the tagger-generated positive prompt.

`archive_path` and `lora_root` are paths that should be given in the context of the system where you're running the `post_to_api.py` program. This is important to keep in mind if the `post_to_api.py` program and ComfyUI are running on different machines, or different virtual machines.

If you have these things set, for example in your configuration file, you can just provide the path of your workflow as the first argument, then the folder (relative to the `archive_path` where to read the first images from), and then as many secondary paths as you would like to generate into and then read from in succession. The saving node the workflow uses will create these folders if they don't already exist. See the Example section above for an illustration of how to invoke it.

### optional configuration

There are a number of other flags that can be adjusted but are not required. Again, these can be set from the command line or the configuration file.

* `comfy_address` if your Comfy instance is bound on a different IP address to 127.0.0.1, you can provide it here.
* `comfy_on_windows` if True, this will replace some forward slashes in particular settings with backslashes.
* `comfy_port` allows you to override the default port of 8188.
* `lora_prefix` allows you to limit the loras found in the `lora_root` folder to those taken from a specific subfolder - useful if you keep, for example, style loras in their own location.
* `output_path` if your Comfy install is on a different machine or container, you might have two different ways of addressing the same location. `output_path` is given relative to the standard Comfy output folder, and sets where the image will be saved to - if it is not set, the `archive_path` will be used instead. An example might make this clearer. Imagine Comfy is running on a windows machine, with an output folder located on E:\Comfy\output. A user wants to run this script from a linux machine on the same network, with E:\Comfy\ available as a network drive on /home/user/usrmnt/comfy/. For the user, `archive_path` should be /home/user/usrmnt/comfy/output/, but the `output_path` should just be an empty string. I am still not entirely happy with this setup and may need to revisit it.

* `disable_llm` if True, removes the LLaVA related checks and workflow components.
* `blank_loras` should be an integer. For a value _n_, it stuffs _n_ "fake" slots into the LoRA array. If you have two LoRA in your folder, and set `blank_loras` to 2, you will have a 50% chance of a LoRA being applied to your image.
* `save_predetail` if True, saves both the finished image and the image prior to face detailing in the output folder. If you are running a chain of generations, this will lead to the number of images doubling at each step, so look out.
* `cfg` and `steps` do what you'd expect, can be helpful when switching the checkpoint on the fly.
* `queue_poll_delay` should be a float - how often the program tests to see if the queue has emptied before submitting the next jobs. It waits for an empty queue so that images generated in earlier runs are available for use in later ones. The default is 0.25 seconds.

## other notes

It is useful (for me) to be able to easily compare images in the same lineage. This script tries to promote this with how it names the output files. If the input file has a filename starting with four digits, or an X followed by four digits, those same digits will be used on the output of the process. If the file does _not_ have a name matching this pattern, the program will attempt to assign one (although this may have mixed results where some files have this prefix and some do not.) You can then use a program like [Diffusion.Toolkit](https://github.com/RupertAvery/DiffusionToolkit) to filter to a specific parent folder and then sort by file name to bring all the relevant items together.