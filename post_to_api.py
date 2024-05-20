from urllib import request
from PIL import Image
import argparse
import base64
import copy
import datetime
import io
import json
import mimetypes
import os
import random
import re
import sys
import tempfile
import time
import tomlkit

class ConfigManager:
    def __init__(self, with_toml = 'default.toml'):
        with open(with_toml) as fp:
            self._state = tomlkit.load(fp)

    # set_key('latent.landscape', [1216, 832])    
    def set(self, key, value):
        address = key.split('.')
        current = self._state
        for part in address[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[address[-1]] = value
    
    # get_key('wd14.banned', [])
    def get(self, key, default = None, collapse = False):
        address = key.split('.')
        current = self._state
        for part in address:
            if part not in current:
                return default
            current = current[part]
        if collapse and isinstance(current, list):
            current = random.choice(current)
        return current

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

# when we go to name the new file, if it starts with digits and an 
# underscore, (optional X prefix) we will provide the same digits
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

def get_queue_length():
    address = configuration.get('server.address', '127.0.0.1')
    port = configuration.get('server.port', 8188)
    req = request.Request(f'http://{address}:{port}/queue')
    res = request.urlopen(req)
    result = json.load(res)
    total_jobs = len(result.get('queue_running', [])) + len(result.get('queue_pending', []))
    return total_jobs

def submit_workflow(wf):
    address = configuration.get('server.address', '127.0.0.1')
    port = configuration.get('server.port', 8188)
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
    
def action_workflow(wf, f_d, output_prefix, dump = False):
    if dump:
        dump_fn = re.sub(r'[\\/*?:"<>|]', '_', f'{f_d}_{output_prefix}_workflow_')
        with tempfile.NamedTemporaryFile(dir = '.', prefix = dump_fn, suffix = '.json', delete = False, mode = 'w') as file:
            json.dump(wf, file, indent = 4)
    else:
        rs = submit_workflow(wf)

def template_workflow(workflow_template, config, image_path, destination, prefix):
    wf = copy.deepcopy(workflow_template)

    if config.get('server.on_windows', False): fix_slashes = lambda x: x.replace('/', '\\')
    else: fix_slashes = lambda x: x

    # seed
    wf['seed']['inputs']['seed'] = random.randint(0, 1125899906842623)

    # output path
    archive_root = config.get('paths.archive', '', True)
    output_root = config.get('paths.output', archive_root, True)
    wf['full_path']['inputs']['string'] = fix_slashes(os.path.join(output_root, destination))
    wf['save_image']['inputs']['filename'] = f'{prefix}_%time_%basemodelname_%seed'

    # do we save both images or only the detailed one?
    if config.get('misc.save_predetail', False, True):
        wf['batch_images']['inputs'] = {
            "images_a": ["ksampler", 3],
            "images_b": ["face_detailer", 0 ]
        }
    else:
        wf['batch_images']['inputs'] = {
            "images_a": ["face_detailer", 0 ]
        }

    # is llava tagging on or off?
    if config.get('llava.disable', False):
        if 'llava_tagger' in wf:
            del wf['llava_tagger']
            wf['combine_prompt_and_preamble']['inputs']['text_c'] = ""
    else:
        wf['llava_tagger']['inputs']['prompt'] = config.get('llava.prompt', 'Please describe this image in detail.', True)
        wf['llava_tagger']['inputs']['model'] = fix_slashes(config.get('llava.model', 'llama/llava-v1.5-7b-Q4_K', True))
        wf['llava_tagger']['inputs']['mm_proj'] = fix_slashes(config.get('llava.projector', 'llama/llava-v1.5-7b-mmproj-Q4_0', True))

    # prompt
    wf['preamble']['inputs']['string'] = config.get('prompt.preamble', '', True)
    wf['negative_prompt']['inputs']['string'] = config.get('prompt.negative', '', True)

    # wd14 tagger
    banned_tags = ', '.join(config.get('wd14.banned', [], False))
    wf['wd14_tagger']['inputs']['exclude_tags'] = banned_tags

    # model, 
    checkpoint = fix_slashes(config.get('model.checkpoint', 'ponyFaetality_v10.safetensors', True))
    if 'base_ckpt_name' in wf['load_model']['inputs']: wf['load_model']['inputs']['base_ckpt_name'] = checkpoint
    if 'ckpt_name' in wf['load_model']['inputs']: wf['load_model']['inputs']['ckpt_name'] = checkpoint
    
    # cfg and steps
    cfg = config.get('model.cfg', None)
    if cfg: wf['ksampler']['inputs']['cfg'] = cfg
    steps = config.get('model.steps', None)
    if steps: wf['ksampler']['inputs']['steps'] = steps

    # pack image
    data_url, img_size = image_to_data_url(image_path)
    wf['image_loader']['inputs']['image_data'] = data_url

    # latent size
    img_width, img_height = img_size
    if img_width == img_height:
        lw, lh = config.get('latent.square', [1024, 1024], False)
    elif img_width > img_height:
        lw, lh = config.get('latent.landscape', [1216, 832], False)
    else:
        lw, lh = config.get('latent.portrait', [832, 1216], False)
    wf['load_model']['inputs']['empty_latent_width'] = lw
    wf['load_model']['inputs']['empty_latent_height'] = lh

    def processLoraRecord(lr):
        name = lr.get('name', 'None')
        if name == 'None':
            return ('None', 0, 0, '', '')
        global_strength = lr.get('strength', 1.0)
        model_strength = lr.get('model_strength', global_strength)
        clip_strength = lr.get('clip_strength', global_strength)
        trigger = lr.get('trigger', '')
        neg_trigger = lr.get('neg_trigger', '')
        return (name, model_strength, clip_strength, trigger, neg_trigger)

    lora_used = []

    lora_choices = config.get('lora', [{'name': 'None'}], False)
    ln, lm, lc, lt, ltn = processLoraRecord(random.choice(lora_choices))
    wf['lora_stacker']['inputs']['lora_name_1'] = fix_slashes(ln)
    if ln != 'None':
        lora_used.append(ln)
    wf['lora_stacker']['inputs']['model_str_1'] = lm
    wf['lora_stacker']['inputs']['clip_str_1'] = lc
    if lt: wf['preamble']['inputs']['string'] += f', {lt}'
    if ltn: wf['negative_prompt']['inputs']['string'] += f', {ltn}'
    
    lindex = 2
    forced_loras = config.get('model.force_lora', [], False)
    for forced_lora in forced_loras:
        ln, lm, lc, lt, ltn = processLoraRecord(forced_lora)
        wf['lora_stacker']['inputs'][f'lora_name_{lindex}'] = fix_slashes(ln)
        if ln != 'None':
            lora_used.append(ln)
        wf['lora_stacker']['inputs'][f'model_str_{lindex}'] = lm
        wf['lora_stacker']['inputs'][f'clip_str_{lindex}'] = lc
        if lt: wf['preamble']['inputs']['string'] += f', {lt}'
        if ltn: wf['negative_prompt']['inputs']['string'] += f', {ltn}'
        wf['lora_stacker']['inputs']['lora_count'] = lindex
        lindex += 1

    lora_used.sort()    
    lora_sort_key = "!".join(lora_used)

    overload_replace = config.get('overload.replace', None, True)
    if overload_replace:
        wf.update(overload_replace)
    
    def deepMerge(dst, src):
        for k in src:
            if k in dst:
                if isinstance(dst[k], dict) and isinstance(src[k], dict):
                    deepMerge(dst[k], src[k])
                else:
                    dst[k] = src[k]
            else:
                dst[k] = src[k]
        return dst
    
    overload_merge = config.get('overload.merge', None, True)
    if overload_merge:
        deepMerge(wf, overload_merge)
    
    # finally, patch the save node, accounting for any changes made by the overloads
    wf['save_image']['inputs']['steps'] = wf['ksampler']['inputs']['steps']
    wf['save_image']['inputs']['cfg'] = wf['ksampler']['inputs']['cfg']
    wf['save_image']['inputs']['modelname'] = wf['ksampler']['inputs']['cfg']
    if 'base_ckpt_name' in wf['load_model']['inputs']: wf['save_image']['inputs']['modelname'] = wf['load_model']['inputs']['base_ckpt_name']
    if 'ckpt_name' in wf['load_model']['inputs']: wf['save_image']['inputs']['modelname'] = wf['load_model']['inputs']['ckpt_name']
    wf['save_image']['inputs']['sampler_name'] = wf['ksampler']['inputs']['sampler_name']
    wf['save_image']['inputs']['scheduler'] = wf['ksampler']['inputs']['scheduler']

    # last tidy
    if wf['preamble']['inputs']['string'].startswith(', '): wf['preamble']['inputs']['string'] = wf['preamble']['inputs']['string'][2:]
    if wf['negative_prompt']['inputs']['string'].startswith(', '): wf['negative_prompt']['inputs']['string'] = wf['negative_prompt']['inputs']['string'][2:]

    return (checkpoint, lora_sort_key, wf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('workflow', type=lambda x: is_valid_file(parser, x))
    parser.add_argument('folder_f', type=str)
    parser.add_argument('folder_t', nargs="+", type=str)
    parser.add_argument('--config_file', type=lambda x: is_valid_file(parser, x), default='default.toml')
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--sort', action='store_true')
    args = parser.parse_args()

    configuration = ConfigManager(args.config_file)

    # we originally had a fairly rich system for command line overrides here, but it seemed like
    # more trouble than it was worth. during the switch to toml, it was decided to get rid of it
    #
    # if you need oneshot changes, it probably is genuinely easier to copy your existing config,
    # add an [override] section as described in the README and use that

    necessaries = [
        'paths.archive',
        'model.checkpoint',
    ]

    for necessary_key in necessaries:
        available = configuration.get(necessary_key, None, True)
        if not available:
            print(f'Necessary configuration key \'{necessary_key}\' must be provided in {args.config_file}. Exiting.')
            sys.exit(1)

    # NAUGHTY, should define a getter for this
    print('Current configuration:', configuration._state)
    if args.dump:
        print('Processed workflows will be saved to disk, but not submitted.')

    with open(args.workflow, 'rt', encoding = 'utf-8') as fh:
        master_workflow_object = json.load(fh)
    if 'comment' in master_workflow_object:
        print('Workflow comment was:', master_workflow_object['comment']['inputs']['string'])
        del master_workflow_object['comment']

    # create our pairs of folders
    folder_list = [args.folder_f] + args.folder_t
    pairings = zip(folder_list[:-1], folder_list[1:])

    stored_workflows = []
    for pair_index, pair in enumerate(pairings):
        f_s, f_d = pair
        print(f'** standing by to run workflow from {f_s} to {f_d}')
        while(get_queue_length() > 0):
            time.sleep(configuration.get('server.poll_delay', 0.25, True))
        print(f'   queue has emptied, time is {datetime.datetime.now()}')

        input_folder = os.path.join(configuration.get('paths.archive', '', True), f_s)
        print(f'   enumerating images in {input_folder}')
        input_images = []
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    input_images.append(os.path.join(root, file))
        print(f'   found {len(input_images)} files')
        print('   ', end = '')
        sys.stdout.flush()

        for image_index, image_path in enumerate(input_images):
            output_prefix = extract_prefix(image_path)
            csort, lsort, templated_workflow = template_workflow(
                master_workflow_object,
                configuration,
                image_path,
                f_d,
                output_prefix
            )
            if args.sort:
                stored_workflows.append(((csort, lsort), templated_workflow, f_d, output_prefix, args.dump))
            else:
                action_workflow(templated_workflow, f_d, output_prefix, args.dump)
            print('.', end = '')
            sys.stdout.flush()
            if not args.dump:
                time.sleep(configuration.get('server.poll_delay', 0.25, True))

        print(f' - complete')
        if args.sort:
            print('  now sorting workflows to minimise model loads')
            stored_workflows.sort(key = lambda x: x[0])
            if args.dump:
                print('  dumping workflows')
            else:
                print('  submitting workflows')
            for _, wf, f_d, op, d in stored_workflows:
                action_workflow(wf, f_d, op, d)
                if not args.dump:
                    time.sleep(configuration.get('server.poll_delay', 0.25, True))
