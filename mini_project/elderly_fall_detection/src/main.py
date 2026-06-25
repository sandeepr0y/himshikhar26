import os
import cv2
from ultralytics import YOLO

from utils import get_app_root, slice_video, get_body_angle, get_box_ratio


RAW_VIDEOS = [
    os.sep.join(('data', 'raw', 'fall',)),
    os.sep.join(('data', 'raw', 'adl',)),
]


class ElderlyFallDetection(object):

    def __init__(self, app_root):
        self._app_root = app_root
        self._model_path = app_root / 'models' / 'yolo26m-pose.pt'
        self._model = None
        self._person_track = {}
    
    @property
    def model(self):
        if self._model is None:
            self._model = YOLO(self._model_path)
        return self._model
    
    def pre_process_raw_videos(self):
        for dir_path in RAW_VIDEOS:
            video_dir = self._app_root / dir_path
            imediate_dir = str(video_dir).split(os.sep)[-1]
            for f in os.listdir(self._app_root / dir_path):
                video_file_path = video_dir / f
                output_file_path = self._app_root / os.sep.join(('data', 'rgb', imediate_dir,)) / f
                slice_video(video_file_path, output_file_path)
    
    def __process_frame(self, frame, box, track_id, person_kp):
        bb_x1, bb_y1, bb_x2, bb_y2 = box

        # Extract Left (Index 5) and Right (Index 6) shoulders
        left_shoulder = person_kp[5]   # [x, y]
        right_shoulder = person_kp[6]  # [x, y]
        left_hip = person_kp[11]       # [x, y]
        right_hip = person_kp[12]      # [x, y]
        left_knee = person_kp[13]
        right_knee = person_kp[14]
        left_ankle = person_kp[15]
        right_ankle = person_kp[16]

        # Unpack coordinates
        ls_x, ls_y = int(left_shoulder[0]), int(left_shoulder[1])
        rs_x, rs_y = int(right_shoulder[0]), int(right_shoulder[1])
        lh_x, lh_y = int(left_hip[0]), int(left_hip[1])
        rh_x, rh_y = int(right_hip[0]), int(right_hip[1])
        lk_x, lk_y = int(left_knee[0]), int(left_knee[1])
        rk_x, rk_y = int(right_knee[0]), int(right_knee[1])
        la_x, la_y = int(left_ankle[0]), int(left_ankle[1])
        ra_x, ra_y = int(right_ankle[0]), int(right_ankle[1])

        body_angle = get_body_angle(
            (ls_x, ls_y, lh_x, lh_y,),
            (rs_x, rs_y, rh_x, rh_y,),
        )

        box_ratio = get_box_ratio(bb_x2 - bb_x1, bb_y2 - bb_y1)
        red = (0, 0, 255)
        blue = (255, 0, 0)
        green = (0, 255, 0)
        black = (0, 0, 0)
        is_lying = False

        if body_angle > 65 and box_ratio < 1:
            status = "Lying Down"
            text_color = red
            is_lying = True
        else:
            shoulder_y = ls_y if ls_y > 0 else rs_y
            if shoulder_y > 0:
                for _y in (lk_y, la_y, rk_y, ra_y,):
                    if _y > 0 and _y <= shoulder_y:
                        is_lying = True
                        break
        
        if is_lying:
            status = "Lying Down"
            text_color = red
        elif box_ratio < 1:
            if track_id in self._person_track:
                is_lying = self._person_track[track_id]['is_lying']
                if is_lying:
                    status = "Lying Down"
                    text_color = red
                else:
                    status = "Standing"
                    text_color = green
            else:
                # give preference to box-ratio
                status = "Lying Down"
                text_color = red
                is_lying = True
        else:
            status = "Standing"
            text_color = green
        
        self._person_track[track_id] = {
            'body_angle': body_angle,
            'box_ratio': box_ratio,
            'is_lying': is_lying
        }

        # if body_angle < 0:
        #     status = "Unknown"
        #     text_color = (255, 0, 0) # Blue for Unknown
        # if body_angle > 65:
        #     status = "Lying Down"
        #     text_color = (0, 0, 255) # Red for danger/lying down
        # else:
        #     status = "Standing"
        #     text_color = (0, 255, 0) # Green for standing

        # Only draw/process if the keypoints are actually detected (not 0,0)
        if ls_x > 0 and ls_y > 0:
            cv2.circle(frame, (ls_x, ls_y), 2, red, -1) # Red circle on Left Shoulder
            
        if rs_x > 0 and rs_y > 0:
            cv2.circle(frame, (rs_x, rs_y), 2, blue, -1) # Blue circle on Right Shoulder
        
        if lh_x > 0 and lh_y > 0:
            cv2.circle(frame, (lh_x, lh_y), 2, red, -1) # Red circle on Left Hip
        
        if rh_x > 0 and rh_y > 0:
            cv2.circle(frame, (rh_x, rh_y), 2, blue, -1) # Blue circle on Right Hip

        if lk_x > 0 and lk_y > 0:
            cv2.circle(frame, (lk_x, lk_y), 2, black, -1) # Black circle on Left Knee
        
        if rk_x > 0 and rk_y > 0:
            cv2.circle(frame, (rk_x, rk_y), 2, black, -1) # Black circle on Right Knee

        if la_x > 0 and la_y > 0:
            cv2.circle(frame, (la_x, la_y), 2, black, -1) # Black circle on Left Ankle
        
        if ra_x > 0 and ra_y > 0:
            cv2.circle(frame, (ra_x, ra_y), 2, black, -1) # Black circle on Right Ankle
        
        cv2.putText(frame, f"{status}, {body_angle:.2f}, {box_ratio:.2f}", (bb_x1, bb_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)

        return frame
    
    def track(self):
        # video_file = self._app_root / os.sep.join(('data', 'rgb', 'fall',)) / 'fall-01-cam0.mp4'
        # video_file = self._app_root / os.sep.join(('data', 'rgb', 'fall',)) / 'fall-04-cam0.mp4'
        video_file = self._app_root / os.sep.join(('data', 'rgb', 'adl',)) / 'adl-01-cam0.mp4'
        cap = cv2.VideoCapture(video_file)

        # Get properties of the original video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        # output_path = self._app_root / 'data' / 'detect' / 'fall-01-cam0.mp4'
        # output_path = self._app_root / 'data' / 'detect' / 'fall-04-cam0.mp4'
        output_path = self._app_root / 'data' / 'detect' / 'adl-01-cam0.mp4'
        if os.path.exists(output_path):
            os.remove(output_path)
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(frame, persist=True, save=False, verbose=False, device=0, classes=0)

            if results[0].boxes.id is not None and results[0].keypoints is not None:

                boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Using .int() maps directly to standard ints
                track_ids = results[0].boxes.id.int().cpu().tolist()
                # class_indices = results[0].boxes.cls.int().cpu().tolist()
                # confidences = results[0].boxes.conf.cpu().tolist()
                keypoints_data = results[0].keypoints.xy.cpu().numpy()

                for box, track_id, person_kp in zip(boxes, track_ids, keypoints_data):
                    frame = self.__process_frame(frame, box, track_id, person_kp)
            
            # Write the annotated frame to the output video
            out.write(frame)
        
        cap.release()
        out.release()

        print(f"Successfully saved annotated video to: {output_path}")

    @classmethod
    def main(cls):
        obj = cls(get_app_root())
        # obj.pre_process_raw_videos()
        obj.track()


if __name__ == '__main__':
    ElderlyFallDetection.main()
