import cv2
import numpy as np

def save_color_histogram_png(image_path, out_path="hist.png", h=400, w=512):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    colors = [(255,0,0), (0,255,0), (0,0,255)]  # BGR for drawing

    for ch, col in enumerate(colors):
        hist = cv2.calcHist([img], [ch], None, [256], [0,256]).flatten()
        hist = hist / (hist.max() + 1e-9)  # normalize 0..1

        for x in range(256):
            x0 = int(x * (w-1) / 255)
            y0 = h - 1
            y1 = int((h - 1) * (1 - hist[x]))
            cv2.line(canvas, (x0, y0), (x0, y1), col, 1)

    cv2.imwrite(out_path, canvas)
    print("Saved:", out_path)

save_color_histogram_png("C:/Users/sriva/Downloads/2_curve_snr1.6_level16.png",
                         "C:/Users/sriva/Downloads/opt_hist.png")
