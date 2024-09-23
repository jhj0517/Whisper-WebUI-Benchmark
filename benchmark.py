import math
import os
import random
from argparse import ArgumentParser
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import *
import gradio as gr
from dataclasses import dataclass, asdict
import copy

import editdistance

from dataset import *
from engine import *
from normalizer import Normalizer


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Whisper-WebUI'))
from modules.whisper.faster_whisper_inference import FasterWhisperInference
from modules.whisper.whisper_base import WhisperBase
from modules.whisper.whisper_parameter import WhisperValues


WorkerResult = namedtuple('WorkerResult', ['num_errors', 'num_words', 'audio_sec', 'process_sec'])
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), "results")

WHISPER_MODEL_DIR = "Whisper-WebUI/models"
FASTER_WHISPER_MODEL_DIR = os.path.join(WHISPER_MODEL_DIR, "faster-whisper")

def _normalize(whisper_result: Dict) -> str:
    transcript = whisper_result[0]
    transcript = [info["text"] for info in transcript]
    transcript = ' '.join(transcript)
    return transcript

def process(
        engine: WhisperWebUIFasterWhisperEngine,
        whisper_params: WhisperValues,
        dataset: Datasets,
        dataset_folder: str,
        indices: Sequence[int],
    ) -> WorkerResult:
    dataset = Dataset.create(dataset, folder=dataset_folder)
    normalizer = Normalizer()

    error_count = 0
    word_count = 0
    for index in indices:
        audio_path, ref_transcript = dataset.get(index)

        transcript = engine.transcribe(audio_path, whisper_params)

        ref_sentence = ref_transcript.strip('\n ').lower()
        ref_words = normalizer.to_american(normalizer.normalize_abbreviations(ref_sentence)).split()
        transcribed_sentence = transcript.strip('\n ').lower()
        transcribed_words = normalizer.to_american(normalizer.normalize_abbreviations(transcribed_sentence)).split()
        print("Ref Words: ", ref_words)
        print("Res Words: ", transcribed_words)

        error_count += editdistance.eval(ref_words, transcribed_words)
        word_count += len(ref_words)

    return WorkerResult(
        num_errors=error_count,
        num_words=word_count,
        audio_sec=engine.audio_sec(),
        process_sec=engine.process_sec())


def main():
    parser = ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--dataset', required=True, choices=[x.value for x in Datasets])
    parser.add_argument('--dataset-folder', required=True)
    parser.add_argument('--aws-profile')
    parser.add_argument('--azure-speech-key')
    parser.add_argument('--azure-speech-location')
    parser.add_argument('--google-application-credentials')
    parser.add_argument('--deepspeech-pbmm')
    parser.add_argument('--deepspeech-scorer')
    parser.add_argument('--picovoice-access-key')
    parser.add_argument('--watson-speech-to-text-api-key')
    parser.add_argument('--watson-speech-to-text-url')
    parser.add_argument('--num-examples', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    engine = WhisperWebUIFasterWhisperEngine(
        model_dir=FASTER_WHISPER_MODEL_DIR,
    )
    whisper_params = WhisperValues(
        model_size="large-v2",
        beam_size=5,
        best_of=5,
        compute_type="float16",
    )
    whisper_params_w_vad = WhisperValues(
        model_size="large-v2",
        beam_size=5,
        best_of=5,
        compute_type="float16",
        # VAD
        vad_filter=True,
        threshold=0.5,
        min_silence_duration_ms=1000,
        speech_pad_ms = 1000
    )
    whisper_params_w_vad_bgm_separation = WhisperValues(
        model_size="large-v2",
        beam_size=5,
        best_of=5,
        compute_type="float16",
        # VAD
        vad_filter=True,
        threshold=0.5,
        min_silence_duration_ms=1000,
        speech_pad_ms=1000,
        # BGM Separation (MDX model)
        is_bgm_separate=True,
        uvr_model_size="UVR-MDX-NET-Inst_HQ_4"
    )

    dataset_type = Datasets(args.dataset)
    dataset_folder = args.dataset_folder
    num_examples = args.num_examples
    num_workers = args.num_workers

    engine_params = dict()
    if engine is Engines.AMAZON_TRANSCRIBE:
        if args.aws_profile is None:
            raise ValueError("`aws-profile` is required")
        os.environ['AWS_PROFILE'] = args.aws_profile
    elif engine is Engines.AZURE_SPEECH_TO_TEXT:
        if args.azure_speech_key is None or args.azure_speech_location is None:
            raise ValueError("`azure-speech-key` and `azure-speech-location` are required")
        engine_params['azure_speech_key'] = args.azure_speech_key
        engine_params['azure_speech_location'] = args.azure_speech_location
    elif engine is Engines.GOOGLE_SPEECH_TO_TEXT or engine == Engines.GOOGLE_SPEECH_TO_TEXT_ENHANCED:
        if args.google_application_credentials is None:
            raise ValueError("`google-application-credentials` is required")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.google_application_credentials
    elif engine is Engines.PICOVOICE_CHEETAH:
        if args.picovoice_access_key is None:
            raise ValueError("`picovoice-access-key` is required")
        engine_params['access_key'] = args.picovoice_access_key
    elif engine is Engines.PICOVOICE_LEOPARD:
        if args.picovoice_access_key is None:
            raise ValueError("`picovoice-access-key` is required")
        engine_params['access_key'] = args.picovoice_access_key
    elif engine is Engines.IBM_WATSON_SPEECH_TO_TEXT:
        if args.watson_speech_to_text_api_key is None or args.watson_speech_to_text_url is None:
            raise ValueError("`watson-speech-to-text-api-key` and `watson-speech-to-text-url` are required")
        engine_params['watson_speech_to_text_api_key'] = args.watson_speech_to_text_api_key
        engine_params['watson_speech_to_text_url'] = args.watson_speech_to_text_url
    else:
        print(f"Transcript via {engine.__class__.__name__}:")

    dataset = Dataset.create(dataset_type, folder=dataset_folder)
    indices = list(range(dataset.size()))
    print(f"Dataset size: {len(indices)}")

    res = process(
        engine=engine,
        whisper_params=whisper_params_w_vad,
        dataset=dataset_type,
        dataset_folder=dataset_folder,
        indices=indices[:],
    )
    num_errors = res.num_errors
    num_words = res.num_words

    rtf = res.process_sec / res.audio_sec
    word_error_rate = 100 * float(num_errors) / num_words

    print(f'WER: {word_error_rate:.2f}')
    print(f'RTF: {rtf}')
    print(f'NUM_ERROR: {num_errors}')
    print(f'NUM_WORDS: {num_words}')
    print(f'PROCESS_SEC: {res.process_sec}')
    print(f'AUDIO_SEC: {res.audio_sec}')

    results_log_path = os.path.join(RESULTS_FOLDER, dataset_type.value, f"{str(engine)}_w-vad-uvr.log")
    os.makedirs(os.path.dirname(results_log_path), exist_ok=True)
    with open(results_log_path, "w") as f:
        f.write(f"WER: {str(word_error_rate)}\n")
        f.write(f"RTF: {str(rtf)}\n")
        f.write(f"DATASET: {dataset_type.value}\n")
        f.write(f"NUM_ERROR: {str(num_errors)}\n")
        f.write(f"NUM_WORDS: {str(num_words)}\n")
        f.write(f"PROCESS_SEC: {res.process_sec}\n")
        f.write(f"AUDIO_SEC: {res.audio_sec}\n")



if __name__ == '__main__':
    main()
