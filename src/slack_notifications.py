import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
load_dotenv()

# Environment variables
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
channel_id = os.getenv("SLACK_CHANNEL_ID") or "C091MC23QTS"

slack_client = WebClient(token=SLACK_BOT_TOKEN)

def send_slack_alert_summary(total_count, top_frames):
    try:
        summary_text = f":rotating_light: {total_count} distress frames detected. Top results below:"
        slack_client.chat_postMessage(channel=channel_id, text=summary_text)

        for i, (conf, image_path) in enumerate(top_frames):
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                continue

            with open(image_path, "rb") as f:
                response = slack_client.files_upload_v2(
                    channel=channel_id,
                    filename=f"frame_{i+1}.jpg",
                    file=f,
                    title=f"Frame #{i+1}",
                    initial_comment=f"📸 Frame #{i+1} | Confidence: {conf:.2f}"
                )
                if not response["ok"]:
                    print(f"⚠️ Slack upload error: {response}")
    except SlackApiError as e:
        print(f"❌ Slack upload failed: {e.response['error']}")
