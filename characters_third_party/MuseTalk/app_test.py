import os
import sys
import json
import glob
import time
import copy
import torch
import pickle
import shutil
import threading
import subprocess
import numpy as np
import cv2
from tqdm import tqdm

# import your original utilities (ensure pythonpath includes project root)
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.audio_processor import AudioProcessor

# small web server
from flask import Flask, send_from_directory, jsonify, render_template_string, request

# -------------------------
# CONFIGS (same as before)
# -------------------------
cfg = {
    "avator_1": {
        "preparation": False,
        "bbox_shift": 5,
        "video_path": "data/video/ljj2.mp4",
        "audio_clips": {
            "audio_1": "data/audio/eng.wav"
        }
    }
}

CONFIG = {
    "version": "v15",
    "ffmpeg_path": "./ffmpeg-4.4-amd64-static/",
    "gpu_id": 0,
    "vae_type": "sd-vae",
    "unet_config": "./models/musetalkV15/musetalk.json",
    "unet_model_path": "./models/musetalkV15/unet.pth",
    "whisper_dir": "./models/whisper",
    "inference_config": "./configs/inference/realtime.yaml",
    "bbox_shift": 0,
    "result_dir": "./results/realtime",
    "extra_margin": 10,
    "fps": 25,
    "audio_padding_length_left": 2,
    "audio_padding_length_right": 2,
    "batch_size": 20,
    "output_vid_name": None,
    "parsing_mode": "jaw",
    "left_cheek_width": 90,
    "right_cheek_width": 90,
    "skip_save_images": False
}

# -------------------------
# helpers
# -------------------------
def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)

# -------------------------
# Avatar class (minimal changes)
# -------------------------
@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation, auto_recreate=False):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.preparation = preparation
        self.batch_size = batch_size
        self.auto_recreate = auto_recreate

        version = CONFIG["version"]
        if version == "v15":
            self.base_path = f"./results/{version}/avatars/{avatar_id}"
        else:
            self.base_path = f"./results/avatars/{avatar_id}"

        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": version
        }
        self.idx = 0

        if self.preparation:
            if os.path.exists(self.avatar_path):
                if self.auto_recreate:
                    shutil.rmtree(self.avatar_path)
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    try:
                        self.load_existing()
                    except Exception as e:
                        print("Existing avatar incomplete, re-preparing:", e)
                        shutil.rmtree(self.avatar_path)
                        osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                        self.prepare_material()
            else:
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation=True")
                sys.exit()
            self.load_existing()

    def load_existing(self):
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')),
                                 key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        version = CONFIG["version"]
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path)
        else:
            print(f"copying files from {self.video_path}")
            for filename in sorted(os.listdir(self.video_path)):
                if filename.endswith(".png"):
                    shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")

        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)

        input_latent_list = []
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if version == "v15":
                y2 = min(y2 + CONFIG["extra_margin"], frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
            crop_frame = frame[y1:y2, x1:x2]
            resized = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            input_latent_list.append(resized)

        # store frames/coords/latents (we store resized crops; inference thread will convert to latents)
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle, self.mask_list_cycle = [], []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{i:08d}.png", frame)
            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mode = CONFIG["parsing_mode"] if version == "v15" else "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)
            cv2.imwrite(f"{self.mask_out_path}/{i:08d}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        # save resized crops (child thread will convert to latents using loaded vae)
        torch.save(self.input_latent_list_cycle, self.latents_out_path)

# -------------------------
# Threaded inference (uses models loaded in main thread)
# -------------------------
def inference_thread_func(avatar_id, audio_path, out_name, status, device, vae, unet, pe, whisper, audio_processor):
    """
    Runs in a separate thread but uses the vae/unet/pe/whisper objects that were loaded once in the main thread.
    """
    try:
        print(f"[inference-thread] Starting inference for avatar {avatar_id} on {device}")
        base_path = f"./results/{CONFIG['version']}/avatars/{avatar_id}"
        latents_path = os.path.join(base_path, "latents.pt")
        video_out_path = os.path.join(base_path, "vid_output")
        os.makedirs(video_out_path, exist_ok=True)

        # load saved crops/latents (we saved resized crops during prepare)
        input_latent_list_cycle = torch.load(latents_path)

        # convert stored crops (numpy arrays) to latents using already-loaded vae
        new_latents = []
        for crop_img in input_latent_list_cycle:
            if isinstance(crop_img, np.ndarray):
                rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
                tensor = torch.nn.functional.interpolate(tensor, size=(256,256), mode='bilinear', align_corners=False)
                tensor = tensor.to(device=device, dtype=vae.vae.dtype)
                # use vae helper to get latents
                lat = vae.get_latents_for_unet(tensor) if hasattr(vae, "get_latents_for_unet") else vae.encode(tensor).latent_dist.mean
                new_latents.append(lat)
            else:
                new_latents.append(crop_img)
        input_latent_list_cycle = new_latents
        #import pdb;pdb.set_trace()
        # build whisper chunks
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=unet.model.dtype)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            unet.model.dtype,
            whisper,
            librosa_length,
            fps=CONFIG["fps"],
            audio_padding_length_left=CONFIG["audio_padding_length_left"],
            audio_padding_length_right=CONFIG["audio_padding_length_right"],
        )

        video_num = len(whisper_chunks)
        gen = datagen(whisper_chunks, input_latent_list_cycle, CONFIG["batch_size"])
        timesteps = torch.tensor([0], device=device)

        tmp_dir = os.path.join(base_path, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        res_frame_idx = 0

        for whisper_batch, latent_batch in tqdm(gen, total=int(np.ceil(float(video_num) / CONFIG["batch_size"]))):
            with torch.no_grad():
                audio_feature_batch = pe(whisper_batch.to(device))
                latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
                recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                # convert to uint8 image and save
                if isinstance(res_frame, torch.Tensor):
                    img = (res_frame.clamp(0,1).permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
                else:
                    img = np.array(res_frame).astype(np.uint8)
                cv2.imwrite(f"{tmp_dir}/{res_frame_idx:08d}.png", img)
                res_frame_idx += 1

        # assemble video and merge audio
        temp_video = os.path.join(base_path, "temp.mp4")
        cmd1 = f"ffmpeg -y -v warning -r {CONFIG['fps']} -f image2 -i {tmp_dir}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_video}"
        os.system(cmd1)
        output_vid = os.path.join(video_out_path, out_name + ".mp4")
        cmd2 = f"ffmpeg -y -v warning -i {audio_path} -i {temp_video} -shortest -c:v copy -c:a aac {output_vid}"
        os.system(cmd2)
        # cleanup
        try:
            os.remove(temp_video)
            shutil.rmtree(tmp_dir)
        except:
            pass

        status['state'] = 'processed'
        status['processed_video'] = output_vid
        status['processed_time'] = time.time()
        print(f"[inference-thread] inference done, saved to {output_vid}")
    except Exception as e:
        print("[inference-thread] inference failed:", e)
        status['state'] = 'error'
        status['error_info'] = str(e)

# -------------------------
# Flask (run in a thread)
# -------------------------
def start_flask_server(shared_status, original_video_path, max_loop_seconds=600, port=5000):
    app = Flask(__name__, static_folder='.')
    app.config["PROPAGATE_EXCEPTIONS"] = False
    app.config["TRAP_HTTP_EXCEPTIONS"] = True
    app.config["DEBUG"] = False
    html = """
    <!doctype html>
    <html>
    <head><meta charset="utf-8"><title>Realtime Avatar Playback</title></head>
    <body>
      <h3>Looping Original Video (will insert processed video when ready)</h3>
      <video id="player" width="640" controls autoplay muted loop></video>
      <p id="info"></p>

    <script>
    const player = document.getElementById('player');
    const info = document.getElementById('info');
    const original = "{{original_video}}";
    let processed = null;
    let processedPlayed = false;
    let loopStart = Date.now();
    const maxMs = {{max_ms}};

    function setSource(src, loop=true, autoplay=true, muted=false){
      player.loop = loop;
      player.muted = muted;
      player.src = src;
      if(autoplay) player.play().catch(e=>console.log(e));
    }

    setSource(original, true, true, true);
    info.innerText = "Playing original. Waiting for processed result...";

    async function pollStatus(){
      try{
        const resp = await fetch('/status');
        const j = await resp.json();
        if(j.state === 'processed' && j.processed_video && !processedPlayed){
            if(processed === null){
                processed = '/video?path=' + encodeURIComponent(j.processed_video);
            }
            setSource(processed, false, true, false);
            info.innerText = "Playing processed video...";
            processedPlayed = true;
            player.onended = function(){
                setSource(original, true, true, true);
                info.innerText = "Returned to original loop.";
            }
        } else if (j.state === 'error') {
            info.innerText = "Worker error: " + (j.error_info || "");
        }
      } catch(e){
        console.log("poll error", e);
      }
    }

    setInterval(function(){
        const elapsed = Date.now() - loopStart;
        if(elapsed > maxMs){
            info.innerText = "Max loop time reached. Stopping.";
            player.pause();
            return;
        }
        pollStatus();
    }, 1000);
    </script>
    </body>
    </html>
    """

    @app.route('/')
    def index():
        html_rendered = render_template_string(html, original_video='/video?path=' + original_video_path, max_ms=int(max_loop_seconds*1000))
        return html_rendered

    @app.route('/video')
    def serve_video():
        p = request.args.get('path', '')
        if not p:
            return "No path", 400
        p = os.path.abspath(p)
        if not os.path.exists(p):
            return "Not found", 404
        d = os.path.dirname(p)
        fn = os.path.basename(p)
        return send_from_directory(d, fn)

    @app.route('/status')
    def status():
        return jsonify(dict(shared_status))

    # run flask (threaded True)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # no multiprocessing.spawn here — we use threads to avoid CUDA reinit
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH ...")
        sep = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{CONFIG['ffmpeg_path']}{sep}{os.environ['PATH']}"

    device = torch.device(f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Prepare face parsing instance (for prepare_material)
    if CONFIG["version"] == "v15":
        fp = FaceParsing(
            left_cheek_width=CONFIG["left_cheek_width"],
            right_cheek_width=CONFIG["right_cheek_width"]
        )
    else:
        fp = FaceParsing()

    # status shared dict (thread-safe simple usage)
    status = {'state': 'preparing', 'processed_video': '', 'processed_time': 0}

    # pick avatar config
    avatar_key = list(cfg.keys())[0]
    avatar_cfg = cfg[avatar_key]
    prep = avatar_cfg["preparation"]
    video_path = avatar_cfg["video_path"]
    bbox_shift = 0 if CONFIG["version"] == "v15" else avatar_cfg["bbox_shift"]

    # prepare avatar materials (this does NOT load heavy models)
    avatar = Avatar(
        avatar_id=avatar_key,
        video_path=video_path,
        bbox_shift=bbox_shift,
        batch_size=CONFIG["batch_size"],
        preparation=prep,
        auto_recreate=True
    )

    status['state'] = 'ready'
    status['ready_time'] = time.time()

    # -------------------------
    # LOAD MODELS ONCE (MAIN THREAD)
    # -------------------------
    print("Loading models into main process (only once)...")
    vae, unet, pe = load_all_model(
        unet_model_path=CONFIG["unet_model_path"],
        vae_type=CONFIG["vae_type"],
        unet_config=CONFIG["unet_config"],
        device=device
    )

    timesteps = torch.tensor([0], device=device)
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    weight_dtype = unet.model.dtype

    # load whisper once
    try:
        from transformers import WhisperModel
        whisper = WhisperModel.from_pretrained(CONFIG["whisper_dir"]).to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
    except Exception as e:
        print("Warning: failed to load whisper:", e)
        whisper = None

    audio_processor = AudioProcessor(feature_extractor_path=CONFIG["whisper_dir"])

    # -------------------------
    # Start Flask in a thread (so it doesn't create new GPU processes)
    # -------------------------
    original_video_abspath = os.path.abspath(avatar_cfg["video_path"])
    flask_thread = threading.Thread(target=start_flask_server, args=(status, original_video_abspath, 600, 5000), daemon=True)
    flask_thread.start()
    print("Flask server started on port 5000. Open http://localhost:5000 in your browser.")

    # -------------------------
    # Start inference thread(s) (use the preloaded models)
    # -------------------------
    status['state'] = 'processing'
    worker_threads = []
    for audio_num, audio_path in avatar_cfg["audio_clips"].items():
        t = threading.Thread(target=inference_thread_func, args=(
            avatar_key, audio_path, audio_num, status, device, vae, unet, pe, whisper, audio_processor
        ), daemon=True)
        t.start()
        worker_threads.append(t)
        # if you only want to process one audio at a time, break here
        # break

    # join worker threads (keep main process alive)
    try:
        for t in worker_threads:
            t.join()
    except KeyboardInterrupt:
        print("Interrupted, exiting...")

    print("All inference threads finished. Server remains running (press Ctrl+C to quit).")
    # keep main thread alive so flask thread remains
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down.")
