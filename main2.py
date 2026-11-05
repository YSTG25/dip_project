import cv2
import ollama
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os

model = YOLO('yolov8n.pt')
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

ANALYZE_EVERY_N_FRAMES = 60
frame_count = 0

scene_narrative = "Waiting for initial scene analysis..."
narrative_history = []

def draw_wrapped_text(img, text, x, y, max_width, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    words = text.split(' ')
    lines, current_line = [], ""

    for word in words:
        test_line = current_line + word + " "
        size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if size[0] > max_width and current_line != "":
            lines.append(current_line)
            current_line = word + " "
        else:
            current_line = test_line
    lines.append(current_line)

    for i, line in enumerate(lines):
        cv2.putText(img, line.strip(), (x, y + i * 25), font, font_scale, color, thickness)

    return y + len(lines) * 25

def analyze_global_scene(frame, detections, results):
    global scene_narrative, narrative_history

    if len(detections) == 0:
        scene_narrative = "No significant objects detected in the scene."
        return

    detected_classes = [results.names[cid] for cid in set(detections.class_id)]
    context_string = ", ".join(detected_classes)

    temp_path = "temp_full_scene.jpg"
    cv2.imwrite(temp_path, frame)

    prompt = (
        f"You are an AI video narrator. The system detected these objects: [{context_string}]. "
        "Describe the current scene, the environment, and what the main subjects are doing. "
        "Provide a detailed 2-3 sentence description."
    )

    try:
        response = ollama.chat(model='moondream', messages=[{
            'role': 'user',
            'content': prompt,
            'images': [temp_path]
        }])
        scene_narrative = response['message']['content']

        narrative_history.append(scene_narrative)

    except Exception as e:
        scene_narrative = f"VLM Error: {e}"

    if os.path.exists(temp_path):
        os.remove(temp_path)

def process_video(source_path, target_path):
    global frame_count, scene_narrative, narrative_history

    video_info = sv.VideoInfo.from_video_path(source_path)
    generator = sv.get_video_frames_generator(source_path)

    PANEL_WIDTH = 400

    target_video_info = sv.VideoInfo(
        width=video_info.width + PANEL_WIDTH,
        height=video_info.height,
        fps=video_info.fps
    )

    print("Starting video processing... Press 'q' to stop early.")

    with sv.VideoSink(target_path, video_info=target_video_info) as sink:
        for frame in generator:
            results = model(frame, device='cpu')[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            frame_count += 1
            if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                analyze_global_scene(frame.copy(), detections, results)

            labels = [f"#{tid} {results.names[cid]}" for cid, tid in zip(detections.class_id, detections.tracker_id)] if detections.tracker_id is not None else []
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            h, w = annotated_frame.shape[:2]
            side_panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)

            cv2.putText(side_panel, "LIVE SCENE NARRATIVE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.line(side_panel, (20, 50), (PANEL_WIDTH-20, 50), (255, 255, 0), 1)
            draw_wrapped_text(side_panel, scene_narrative, 20, 80, PANEL_WIDTH - 40, (255, 255, 255))

            combined_view = np.hstack((annotated_frame, side_panel))

            cv2.imshow("Multimodal Perception System", combined_view)
            sink.write_frame(combined_view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    print("\n--- VIDEO PROCESSING COMPLETE ---")
    if len(narrative_history) > 0:
        print("Compiling chronological data to generate final video summary using LLaVA...\n")

        full_text_log = " ".join(narrative_history)

        summary_prompt = (
            "You are a highly intelligent AI assistant. I will provide you with a sequence of chronological descriptions "
            "taken from a video feed. Based on these descriptions, write a cohesive, comprehensive summary of what happened "
            f"in the entire video from start to finish. Remove redundancies and make it flow perfectly:\n\n{full_text_log}"
        )

        try:
            summary_response = ollama.chat(model='llava', messages=[{
                'role': 'user',
                'content': summary_prompt
            }])

            final_summary_text = summary_response['message']['content']

            print("\n================ FINAL VIDEO SUMMARY (LLaVA) ================")
            print(final_summary_text)
            print("=============================================================")

            output_txt_file = "final_video_summary.txt"
            with open(output_txt_file, "w", encoding="utf-8") as f:
                f.write("MULTIMODAL PERCEPTION SYSTEM - FINAL VIDEO SUMMARY\n")
                f.write("====================================================\n\n")
                f.write(final_summary_text)

            print(f"\n[+] Success: The full summary has been saved to '{output_txt_file}' in your project folder.")

        except Exception as e:
            print(f"Error generating final summary with LLaVA: {e}")
    else:
        print("No narrative data was collected to summarize.")

if __name__ == "__main__":
    INPUT_VIDEO = "input_test.mp4"
    OUTPUT_VIDEO = "output_reasoning_ui.mp4"
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)
