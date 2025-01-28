from arguments import Arguments
import cv2
import os
import numpy as np

from birds_eye_module import BirdsEyeView


def main(opt):
    # Initialize processor
    processor = BirdsEyeView(opt)

    # Video capture
    cap = cv2.VideoCapture(opt.source)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = None
    out = None
    if opt.save:
        output_name = opt.source.split("/")[-1]
        output_name = (
            output_name.split(".")[0] + "_output." + output_name.split(".")[-1]
        )
        output_path = os.path.join(os.getcwd(), "inference/output")
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, output_name)
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            opt.outputfps,
            (w, h),
        )

    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Debugging frame size and type
        print(f"Processing frame {frame_num}: shape={frame.shape}, dtype={frame.dtype}")

        frame = processor.process_frame(frame, frame_num, w, h)

        # Debugging processed frame mean value
        print(f"Processed frame {frame_num}: mean pixel value={np.mean(frame)}")

        if opt.view:
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:  # Quit on 'q' or 'ESC'
                print("Exiting on user request.")
                break

        if opt.save and out:
            print(f"Writing frame {frame_num} to output.")
            out.write(frame)

        frame_num += 1
        print(
            f"\r[Input Video: {opt.source}] [{frame_num}/{frame_count} Frames Processed]",
            end="",
        )

    print("\nProcessing complete.")
    if opt.save and output_path:
        print(f"Output saved at {output_path}")

    cap.release()
    cv2.destroyAllWindows()
    if out:
        out.release()


if __name__ == "__main__":
    opt = Arguments().parse()
    main(opt)
