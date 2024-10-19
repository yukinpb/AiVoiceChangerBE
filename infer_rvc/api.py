import os

from gtts import gTTS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import os
import sys
from dotenv import load_dotenv
from infer.modules.vc.modules import VC

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from scipy.io.wavfile import write
import uuid
import uvicorn

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()


from config import Config

device = "cuda:0"
is_half = False
config = Config(device, is_half)

app = FastAPI()
print("Welcome to the Vietnamese Voice API")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Vietnamese Multi-Voice TTS Beta_v0.4 API"}

def process_audio_file(audio_storage_path, chosen_model_name, semitone=0.0):
    index_file = "assets/weight/{}.index".format(os.path.splitext(chosen_model_name)[0])
    config = Config(device, is_half)
    vc = VC(config)
    vc.get_vc(chosen_model_name + ".pth")
    al, be = vc.vc_single(0, audio_storage_path, semitone, None, "rmvpe", None, None, 0.75, 3, 0, 0.25, 0.33)
    tgt_sr, audio_opt = be
    print("Result:", al)
    return tgt_sr, audio_opt

@app.post("/text2speech/")
def text2speech(text: str = "Đây là văn bản sẽ được đọc",
                rvc_pitch: int = -8,
                locate: str = "en",
                voice_option: str = "Nguyễn Ngọc Ngạn"):
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    section_storage_path = "/app/storage/"
    section_id = str(uuid.uuid4())
    section_cloned_file_path = str(section_storage_path + section_id + "_cloned.wav")

    tts_output = "output.wav"

    tts = gTTS(text=text, lang=locate)
    tts.save(tts_output)

    tgt_sr, audio_opt = process_audio_file(tts_output, voice_option, rvc_pitch)
    write(section_cloned_file_path, tgt_sr, audio_opt)

    extra = "attachment; filename={}".format(section_id + "_cloned.wav")
    headers = {
        "Content-Disposition": extra,
        "Content-Type": "audio/mpeg",
    }

    return FileResponse(section_cloned_file_path, headers=headers)

if __name__ == "__main__":
    uvicorn.run("api:app",
                host=str(os.getenv("api_host")),
                port=int(os.getenv("expose_port")),
                workers=int(os.getenv("api_workers")),
                log_level="info")














