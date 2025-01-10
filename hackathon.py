import asyncio
import os
import requests
from moviepy.editor import VideoFileClip
from pytube import YouTube
from pytubefix import YouTube
from pytubefix.cli import on_progress
import cv2
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import ssl, base64
from ultralytics import YOLO
from PIL import Image

import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.schema import TextNode

model = YOLO("yolo11n.pt")
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1/bin/ffmpeg"
ssl._create_default_https_context = ssl._create_stdlib_context
OPENAI_API_KEY = ""


def create_embeddings(data):
    # hugginface model for embedding
    Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5")

    #pinconeAPI
    os.environ[
        "PINECONE_API_KEY"
    ] = ""

    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)

    # delete if needed
    # pc.delete_index("llamaindex-ragathon-demo-index-v2")

    # creating index for now not needed
    # pc.create_index(
    #     "llamaindex-ragathon-demo-index-v2",
    #     dimension=768, #dimesions for the embedding
    #     metric="euclidean",
    #     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    # )

    pinecone_index = pc.Index("llamaindex-ragathon-demo-index-v2")

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, namespace="Default"
    )

    nodes =[]
    for d in data:
        nodes.append(TextNode(
            text = "Timestamp: " + str(d['timestamp']) + "\n" + 'video_id: ' + str(d['video_id']) + "\n\n" + d['text'],
            metadata={
                "timestamp":d['timestamp'],
                "video_id":d['video_id'],
                "agent": d['agent']
            }
        ))

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

def generate_embedding(agent, video_id, base_path):
    agent_folder_mapping = {
        'image_captioning': 'image_captionings',
        'transcripts': 'transcripts',
        'yolo': 'yolo_outputs'  # Add other agents and their folder names as needed
    }
    
    if agent not in agent_folder_mapping:
        raise ValueError(f"Unknown agent '{agent}'")
    
    path = os.path.join(base_path, 'chunks', agent_folder_mapping[agent])
    data_array = []
    
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            parts = filename.split('_')
            timestamp = parts[1]
            with open(os.path.join(path, filename), 'r') as file:
                text = file.read()
            
            entry = {
                'text': f"Timestamp: {timestamp}\nvideo_id: {video_id}\n\n{text.strip()}",
                'timestamp': timestamp,
                'video_id': video_id,
                'agent': agent
            }
            data_array.append(entry)
    
    create_embeddings(data_array)

class VideoProcessingWorkflow:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.setup_directories()
        
    def setup_directories(self):
        directories = [self.output_dir, 
                       os.path.join(self.output_dir, "videos"),
                       os.path.join(self.output_dir, "audios"),
                       os.path.join(self.output_dir, "transcripts"),
                       os.path.join(self.output_dir, "images")]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
    async def process_subclip(self, start_time, end_time, subclip_path, audio_path):
        try:
            with VideoFileClip(self.video_path).subclip(start_time, end_time) as clip:
                clip.write_videofile(subclip_path, codec='mpeg4')
                clip.audio.write_audiofile(audio_path)
                transcript = query(audio_path)
                transcript_path = os.path.join(self.output_dir, "transcripts", f"transcript_{start_time}_{end_time}.txt")
                with open(transcript_path, 'w') as f:
                    f.write(transcript['text'])
                print(f"Processed and transcribed video and audio from {start_time} to {end_time} seconds.")
                
                # Extract keyframes
                images_dir = os.path.join(self.output_dir, "images", f"video_{start_time}_{end_time}")
                os.makedirs(images_dir, exist_ok=True)
                image_captioning_content = self.extract_keyframes_dl(subclip_path, images_dir)
                print("Final ")
                print(image_captioning_content)

                print(self.output_dir)
                image_captioning_path = os.path.join(self.output_dir, "image_captionings", f"imagecaptioning_{start_time}_{end_time}.txt")

                print(self.output_dir)
                print(image_captioning_path)
                os.makedirs(os.path.dirname(image_captioning_path), exist_ok=True)

                with open(image_captioning_path, 'w') as f:
                    f.write(image_captioning_content)
                print(f"Processed and transcribed video and audio from {start_time} to {end_time} seconds.")
                
        except Exception as e:
            print(f"Error processing media from {start_time} to {end_time} seconds: {str(e)}")

    async def split_video_and_audio(self):
        try:
            with VideoFileClip(self.video_path) as clip:
                duration = int(clip.duration)
                tasks = []
                for start_time in range(0, duration, 20):
                    end_time = min(start_time + 20, duration)
                    subclip_path = os.path.join(self.output_dir, "videos", f"video_{start_time}_{end_time}.mp4")
                    audio_path = os.path.join(self.output_dir, "audios", f"audio_{start_time}_{end_time}.mp3")
                    task = asyncio.create_task(self.process_subclip(start_time, end_time, subclip_path, audio_path))
                    tasks.append(task)
                await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error opening video file: {str(e)}")

    def extract_keyframes_dl(self, video_path, output_dir, threshold=0.5):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove files and links
        print(f"Cleared all contents in {output_dir}")
        
        print("Generating keyframes...")
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()

        if not ret:
            print("Failed to read video")
            cap.release()
            return

        # Prepare the transformation and model for feature extraction
        transform = Compose([
            Resize((224, 224)),  # Resize frames for model input
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()

        frame_count = 0
        prev_features = self.get_frame_features(prev_frame, model, transform)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            features = self.get_frame_features(frame, model, transform)
            similarity = torch.nn.functional.cosine_similarity(prev_features, features, dim=1)

            # Save frame as a keyframe if the similarity is below the threshold
            if similarity.item() < threshold:
                frame_path = os.path.join(output_dir, f"keyframe_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])  # Save with high quality
                print(f"Saved keyframe {frame_count} at {frame_path}")
                prev_features = features  # Update features only if a keyframe is saved

            frame_count += 1

        cap.release()
        folder_path=output_dir
        image_paths = os.listdir(folder_path)
        print(image_paths)

        # Getting the base64 string
        base64_images = [encode_image(os.path.join(folder_path,image_path)) for image_path in image_paths]
        print(len(base64_images))

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        payload = {
        "model": "gpt-4o",
        "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "These are the key frames images which indicate change of scenes when extracted from the set of image frames. Give me an good overall description summary of what is happening in the video so that whenever I search for it in the future, i can know what is happening here."
                    }
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                    for base64_image in base64_images
                ]
            }],
            "max_tokens":300,
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_dict = response.json()
        
        if("choices" in response_dict):      
            content = response_dict['choices'][0]['message']['content']
            return content      
        else:
            print(response_dict)
            raise "Error in response"                  
        
    def get_frame_features(self, frame, model, transform):
        """Extract features from a frame using the specified model and transformation."""
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = transform(frame).unsqueeze(0)
        with torch.no_grad():
            return model(frame_tensor)
        
def download_video(url, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    full_output_path = os.path.join(output_path, 'movie.mp4')

    if os.path.exists(full_output_path):
        os.remove(full_output_path)
        print("Existing file removed.")

    yt = YouTube(url, on_progress_callback=on_progress)
    print(yt.title)
 
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path=output_path, filename='movie.mp4')

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
headers = {"Authorization": "Bearer "}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_yolo_chunks():
    path ='sources/bahubali/chunks/images'
    model = YOLO("yolo11n.pt")
    data =[]
    for filename in os.listdir(path):
        print(filename)
        timestamp = filename.split('_')[1]
        timestamp_end = filename.split('_')[2]

        filepath = path+'/'+filename
        entry = {
                'text': '',
                'timestamp': timestamp,
                'video_id': 'bahubali',
                'agent': 'yolo'  # Replace with the appropriate value if needed
            }
        text ='The yolo objects in the frames from timestamp'+timestamp+'to '+timestamp_end
        for file in os.listdir(filepath):
            # print(file)
            results = model(filepath+'/'+file)

            text += str(results[0].to_json())
            # print(str(results[0].to_json()))
        entry['text'] = text
        # print(entry)
        data.append(entry)
    create_embeddings(data)

if __name__ == "__main__":
    video_id = 'bahubali'
    source = f"sources/{video_id}/"
    video_path = source + "movie.mp4"
    output_dir = source + "chunks"
    
    print("YOUTUBE: Download started")
    download_video('https://www.youtube.com/watch?v=7z1bv8CtQxs', source)
    
    # workflow = VideoProcessingWorkflow(video_path, output_dir)
    # asyncio.run(workflow.split_video_and_audio())
    create_yolo_chunks() # this does pushiing of embeddings also 
    generate_embedding('image_captioning', video_id, source)
    generate_embedding('transcripts', video_id, source)
