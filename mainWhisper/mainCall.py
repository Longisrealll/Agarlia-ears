import whisper
import os
import subprocess

SOURCE = "testFiles\\"
MODEL = "turbo"
ALLMODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
class MainFunction:
    def __init__(self, model):
        if model in ALLMODELS:
            print("Model found, using "+model)
            self.modules = whisper.load_model(model)
        else:
            print(f"Model not found, using turbo as default")
            self.modules = whisper.load_model("turbo")
    
    def transcribe(self, wav_16_file:list[str]) -> dict:

        returnedOutputs={}
        for theFile in wav_16_file:
            wav_file=SOURCE+theFile

            if os.path.exists(wav_file):
                print("----------------------------")
                print("File found ✅")
            else:
                print("----------------------------")
                print("File NOT found ❌")
                return ""
            
            wav_file = self.checkHertz(wav_file)

            audio = whisper.load_audio(wav_file)
            thirtyAudio = whisper.pad_or_trim(audio)

            mel = whisper.log_mel_spectrogram(thirtyAudio, n_mels=self.modules.dims.n_mels).to(self.modules.device)

            _, prob = self.modules.detect_language(mel)
            print(f"Language detected: {max(prob, key=prob.get)}")

            result = self.modules.transcribe(wav_file)
            print("-------------------RESULT-------------------")
            print(result["text"])
            returnedOutputs[theFile] = result['text']
        return returnedOutputs
    
    def checkHertz(self, fileHere:str)->str:
        #idk, this is chat gpt suggestion
        commandCheckHerz=['ffmpeg', '-i', fileHere]
        theFile = fileHere.split(".")[0]
        fixedFile = theFile+"_wav16.wav"

        if os.path.exists(fixedFile):
            return fixedFile

        command=['ffmpeg', '-i', fileHere, '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', fixedFile]

        theHerts = subprocess.run(commandCheckHerz, capture_output=True, text=True)
        sample=theHerts.stderr
        if '16000' not in sample or 'mono' not in sample:
            print("refactoring.........")
            subprocess.run(command, capture_output=True, check=True)
            print("Refactored")
        else:
            print("Good to go")
        
        return fixedFile

    
data = MainFunction("tuijrbo")
data.transcribe(["backGround1.wav", "is30sec.wav", "lessthan30sec.wav", "more30sec.wav"])
