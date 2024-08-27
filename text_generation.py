import whisper
import pyaudio
import queue
import numpy as np
import time
import threading
from typing import Dict, List

text_generation_model = whisper.load_model("small", device="cuda")

audio_queue = queue.Queue()
length_queue = queue.Queue(maxsize=6)


def producer_thread():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=16000,  # 1 second of audio
    )

    print("-" * 80)
    print("Microphone initialized, recording started...")
    print("-" * 80)
    print("TRANSCRIPTION")
    print("-" * 80)

    while True:
        audio_data = b""
        for _ in range(1):
            chunk = stream.read(16000)  # Read 1 second of audio data
            audio_data += chunk

        audio_queue.put(audio_data)


def consumer_thread(stats):
    while True:
        if length_queue.qsize() >= 6:
            with length_queue.mutex:
                length_queue.queue.clear()
                print()

        audio_data = audio_queue.get()
        transcription_start_time = time.time()
        length_queue.put(audio_data)

        # Concatenate audio data in the lenght_queue
        audio_data_to_process = b""
        for i in range(length_queue.qsize()):
            # We index it so it won't get removed
            audio_data_to_process += length_queue.queue[i]

        # convert the bytes data toa  numpy array
        audio_data_array: np.ndarray = (
            np.frombuffer(audio_data_to_process, np.int16).astype(np.float32) / 255.0
        )

        segments, _ = whisper.transcribe(
            model=text_generation_model,
            audio=audio_data_array,
        )

        segments = [s.text for s in segments]

        transcription_end_time = time.time()

        transcription = " ".join(segments)
        # remove anything from the text which is between () or [] --> these are non-verbal background noises/music/etc.
        # transcription = re.sub(r"\[.*\]", "", transcription)
        # transcription = re.sub(r"\(.*\)", "", transcription)
        # # We do this for the more clean visualization (when the next transcription we print would be shorter then the one we printed)
        # transcription = transcription.ljust(MAX_SENTENCE_CHARACTERS, " ")

        transcription_postprocessing_end_time = time.time()

        print(transcription, end="\r", flush=True)

        audio_queue.task_done()

        overall_elapsed_time = (
            transcription_postprocessing_end_time - transcription_start_time
        )
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = (
            transcription_postprocessing_end_time - transcription_end_time
        )
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)


stats: Dict[str, List[float]] = {
    "overall": [],
    "transcription": [],
    "postprocessing": [],
}

producer = threading.Thread(target=producer_thread)
producer.start()

consumer = threading.Thread(target=consumer_thread, args=(stats,))
consumer.start()

try:
    producer.join()
    consumer.join()
except KeyboardInterrupt:
    print("Exiting...")
    # print out the statistics
    print("Number of processed chunks: ", len(stats["overall"]))
    print(
        f"Overall time: avg: {np.mean(stats['overall']):.4f}s, std: {np.std(stats['overall']):.4f}s"
    )
    print(
        f"Transcription time: avg: {np.mean(stats['transcription']):.4f}s, std: {np.std(stats['transcription']):.4f}s"
    )
    print(
        f"Postprocessing time: avg: {np.mean(stats['postprocessing']):.4f}s, std: {np.std(stats['postprocessing']):.4f}s"
    )
    # We need to add the step_in_sec to the latency as we need to wait for that chunk of audio
    print(f"The average latency is {np.mean(stats['overall'])+6:.4f}s")


print(stats.get("transcription"))
exit()
