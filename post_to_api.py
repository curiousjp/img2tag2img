from urllib import request
from PIL import Image
import argparse
import base64
import configparser
import datetime
import io
import json
import mimetypes
import os
import random
import re
import sys
import time

configuration = {}

def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None or not mime_type.startswith('image'):
        raise ValueError("Unsupported file type")
    with Image.open(image_path) as img:
        img_format = mime_type.split('/')[-1].upper()
        # strip metadata
        pixel_data = list(img.getdata())
        new_image = Image.new(img.mode, img.size)
        new_image.putdata(pixel_data)
        with io.BytesIO() as output:
            new_image.save(output, format=img_format)
            image_bytes = output.getvalue()
    base64_bytes = base64.b64encode(image_bytes)
    base64_string = base64_bytes.decode('utf-8')
    data_url = f'data:{mime_type};base64,{base64_string}'
    return data_url, img.size

# when we go to name the new file, if it starts with four digits,
# or an X followed by four digits, we will provide the same digits
# to use as a filename prefix. this lets us use diffusion.toolkit's
# sort_by_name to group all the images with a common lineage together.
#
# if the file doesn't start with either of those patterns, we will
# grab a (nominally) unused number (stored as a function static) and
# use that instead.
#
# this can stuff up in situations where some files in
# the folder have prefixes and others don't.

def extract_prefix(file_path):
    if not hasattr(extract_prefix, "counter"):
        extract_prefix.counter = 0
    filename = os.path.basename(file_path)
    match = re.match(r"^(X?\d+)_", filename)
    if match:
        return match.group(1)  # Return the digits
    extract_prefix.counter += 1
    return f"X{extract_prefix.counter:04}"

def get_style_loras(base_lora_path, subfolder = None, pad_list = 1):
    lora_banlist = [
        'Photo Style LoRA XL.safetensors',
        'hyperrealism_CivitAI.safetensors',
        'RetouchXL_PonyV6_v2.safetensors'
    ]
    candidates = []
    if subfolder:
        folder_path = os.path.join(base_lora_path, subfolder)
    else:
        folder_path = base_lora_path

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.safetensors') and file not in lora_banlist:
                candidates.append(os.path.relpath(os.path.join(root, file), base_lora_path).replace('\\', '/'))
    if pad_list > 0:
        candidates += ([None] * pad_list)
    return candidates

def get_queue_length():
    address = configuration.get('comfy_address', '127.0.0.1')
    port = configuration.get('comfy_port', '8188')
    req = request.Request(f'http://{address}:{port}/queue')
    res = request.urlopen(req)
    result = json.load(res)
    total_jobs = len(result.get('queue_running', [])) + len(result.get('queue_pending', []))
    return total_jobs

def submit_workflow(wf):
    address = configuration.get('comfy_address', '127.0.0.1')
    port = configuration.get('comfy_port', '8188')
    wrapped_workflow = {'prompt': wf}
    wfj = json.dumps(wrapped_workflow).encode('utf-8')
    req = request.Request(f'http://{address}:{port}/prompt', data=wfj)
    res = request.urlopen(req)
    return res

def is_valid_file(p, a):
    if not os.path.exists(a):
        p.error(f'file {a} does not exist!')
    else:
        return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('workflow', type=lambda x: is_valid_file(parser, x))
    parser.add_argument('folder_f', type=str)
    parser.add_argument('folder_t', nargs="+", type=str)
    parser.add_argument('--config_file', type=lambda x: is_valid_file(parser, x), default='default.ini')
    parser.add_argument('--dump', action='store_true')
    args, unknown_arguments = parser.parse_known_args()

    if args.config_file:
        config_object = configparser.ConfigParser()
        config_object.read(args.config_file)
        for section in config_object.sections():
            for k, v in config_object.items(section):
                configuration[k] = v

    # had originally used an iterable here, but there's no
    # really good way to peek these in a for/in loop
    index = 0
    while index < len(unknown_arguments):
        unknown_arg = unknown_arguments[index]
        if index < len(unknown_arguments) - 1:
            next_unknown_arg = unknown_arguments[index + 1]
        else:
            next_unknown_arg = None
        if unknown_arg.startswith('--'):
            key = unknown_arg[2:]
            if '=' in key:
                key, value = key.split('=', 1)
            elif next_unknown_arg == None or next_unknown_arg.startswith('--'):
                value = True
            else:
                value = next_unknown_arg
                # jump over the next argument
                index += 1
            configuration[key] = value
        index += 1

    for k, v in configuration.items():
        if k in ['blank_loras', 'steps', 'latent_long_edge', 'latent_short_edge', 'latent_square_edge']: v = int(v)
        if k in ['queue_poll_delay', 'cfg']: v = float(v)
        if v in ['False', 'false']: v = False
        if v in ['True', 'true']: v = True
        configuration[k] = v

    necessaries = [
        'archive_path',
        'banned_tags',
        'checkpoint',
        'latent_long_edge',
        'latent_short_edge',
        'latent_square_edge',
        'llava_model',
        'llava_projector',
        'llava_prompt',
        'lora_root',
        'negative',
        'preamble'
    ]

    if configuration.get('disable_llm', False):
        necessaries.remove('llava_model')
        necessaries.remove('llava_projector')
        necessaries.remove('llava_prompt')

    for necessary_key in necessaries:
        if necessary_key not in configuration:
            print(f'Necessary configuration key \'{necessary_key}\' must be provided in either {args.config_file} or on the command line. Exiting.')
            sys.exit(1)

    print('Current configuration:', configuration)

    if configuration.get('comfy_on_windows', False):
        fix_slashes = lambda x: x.replace('/', '\\')
    else:
        fix_slashes = lambda x: x

    # create our pairs of folders
    folder_list = [args.folder_f] + args.folder_t
    pairings = zip(folder_list[:-1], folder_list[1:])

    with open(args.workflow, 'rt', encoding = 'utf-8') as fh:
        master_workflow_object = json.load(fh)
    if '_comment' in master_workflow_object:
        print('Workflow comment was:', master_workflow_object['_comment'])
        del master_workflow_object['_comment']

    # overrides
    for k, v in configuration.items():
        if not k.startswith('override-'):
            continue
        pieces = k.split('-')
        # --override-ksampler-sampler_name euler
        if len(pieces) == 3:
            _, node, field = pieces
            type_conversion = None
        # --override-load_model-batch_size-int 3
        if len(pieces) == 4:
            _, node, field, type_conversion = pieces
        match type_conversion:
            case 'str':
                v = str(v)
            case 'int':
                v = int(v)
            case 'float':
                v = float(v)
            case 'wire':
                inwards_node, output_number = v.split(':', 1)
                v = [inwards_node.strip(), int(output_number)]
            case None:
                pass
            case _:
                raise TypeError(f'requested an override with {k}, but no casting case known for {type_conversion}')
        if not node in master_workflow_object:
            raise ValueError(f'requested an override of node {node} via {k}, but no such node in the workflow')
        if not field in master_workflow_object[node]['inputs']:
            raise ValueError(f'requested an override of input {field} on node {node} via {k}, but no such input exists on that node in the workflow')
        master_workflow_object[node]['inputs'][field] = v

    lora_choices = get_style_loras(
        configuration.get('lora_root'),
        configuration.get('lora_prefix'),
        configuration.get('blank_loras', 0)
    )
    print(f'LoRA discovery: {len(lora_choices)} LoRA detected (including \'blank\' LoRA options).')

    if configuration.get('save_predetail', False):
        master_workflow_object['batch_images']['inputs'] = {
            "images_a": ["ksampler", 3],
            "images_b": ["face_detailer", 0 ]
        }
    else:
        master_workflow_object['batch_images']['inputs'] = {
            "images_a": ["face_detailer", 0 ]
        }

    if configuration.get('disable_llm', False):
        if 'llava_tagger' in master_workflow_object:
            del master_workflow_object['llava_tagger']
            master_workflow_object['combine_prompt_and_preamble']['inputs']['text_c'] = ""
    else:
        master_workflow_object['llava_tagger']['inputs']['prompt'] = configuration.get('llava_prompt')
        master_workflow_object['llava_tagger']['inputs']['model'] = fix_slashes(configuration.get('llava_model'))
        master_workflow_object['llava_tagger']['inputs']['mm_proj'] = fix_slashes(configuration.get('llava_projector'))


    master_workflow_object['preamble']['inputs']['string'] = configuration.get('preamble')
    master_workflow_object['negative_prompt']['inputs']['string'] = configuration.get('negative')
    master_workflow_object['wd14_tagger']['inputs']['exclude_tags'] = configuration.get('banned_tags')

    if 'cfg' in configuration:
        master_workflow_object['ksampler']['inputs']['cfg'] = configuration.get('cfg')
    if 'steps' in configuration:
        master_workflow_object['ksampler']['inputs']['steps'] = configuration.get('steps')

    checkpoint = fix_slashes(configuration.get('checkpoint'))
    if 'base_ckpt_name' in master_workflow_object['load_model']['inputs']:
        master_workflow_object['load_model']['inputs']['base_ckpt_name'] = checkpoint
    if 'ckpt_name' in master_workflow_object['load_model']['inputs']:
        master_workflow_object['load_model']['inputs']['ckpt_name'] = checkpoint
    master_workflow_object['save_image']['modelname'] = checkpoint

    for pair in pairings:
        f_s, f_d = pair
        per_directory_wf = dict(master_workflow_object)

        if 'output_path' in configuration:
            output_path = fix_slashes(os.path.join(configuration.get('output_path'), f_d))
        else:
            output_path = fix_slashes(os.path.join(configuration.get('archive_path'), f_d))
        per_directory_wf['full_path']['inputs']['string'] = output_path

        print(f'** standing by to run workflow from {f_s} to {f_d}')
        while(get_queue_length() > 0):
            time.sleep(configuration.get('queue_poll_delay', 0.25))
        print(f'   queue has emptied, time is {datetime.datetime.now()}')

        file_folder = os.path.join(configuration.get('archive_path'), f_s)
        print(f'   enumerating images in {file_folder}')
        input_images = []
        for root, _, files in os.walk(file_folder):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    input_images.append(os.path.join(root, file))

        print(f'   found {len(input_images)} files')
        print('   ', end = '')
        sys.stdout.flush()

        for image_path in input_images:
            wf = dict(per_directory_wf)
            wf['seed']['inputs']['seed'] = random.randint(0, 1125899906842623)
            data_url, img_size = image_to_data_url(image_path)
            wf['image_loader']['inputs']['image_data'] = data_url

            img_width, img_height = img_size
            if img_width == img_height:
                latent_width = configuration.get('latent_square_edge')
                latent_height = latent_width
            elif img_width > img_height:
                latent_width = configuration.get('latent_long_edge')
                latent_height = configuration.get('latent_short_edge')
            else:
                latent_width = configuration.get('latent_short_edge')
                latent_height = configuration.get('latent_long_edge')
            wf['load_model']['inputs']['empty_latent_width'] = latent_width
            wf['load_model']['inputs']['empty_latent_height'] = latent_height

            if not lora_choices:
                lora_choices = [None]
            lora_name = random.choice(lora_choices)
            if lora_name:
                wf['lora_stacker']['inputs']['lora_name_1'] = fix_slashes(lora_name)
                wf['lora_stacker']['inputs']['model_str_1'] = 0.8
                wf['lora_stacker']['inputs']['clip_str_1'] = 0.8
            output_prefix = extract_prefix(image_path)
            wf['save_image']['inputs']['filename'] = f'{output_prefix}_%time_%basemodelname_%seed'
            if args.dump:
                dump_fn = f'{f_d}_{output_prefix}_workflow.json'
                with open(dump_fn, 'w') as file:
                    json.dump(wf, file, indent = 4)
            rs = submit_workflow(wf)
            print('.', end = '')
            sys.stdout.flush()
            time.sleep(0.05)
        print(f' - complete')