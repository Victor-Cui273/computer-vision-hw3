"""Count visible round log ends and pipe openings using HoughCircles."""
from pathlib import Path
import csv
import json
import cv2 as cv
import numpy as np


def detect(image, config):
    scale = config['width'] / image.shape[1]
    preview = cv.resize(image, (config['width'], round(image.shape[0] * scale)))
    x, y, w, h = config['roi'] or (0, 0, preview.shape[1], preview.shape[0])
    stretch = config['stretch']
    crop = cv.resize(preview[y:y+h, x:x+w], (round(w*stretch), h))
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    k = config['blur']
    gray = cv.GaussianBlur(gray, (k, k), 1 if k == 5 else 0.7)
    candidates = []
    for minimum, maximum, threshold in config['bands']:
        circles = cv.HoughCircles(
            gray, cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=config['min_dist'],
            param1=150, param2=threshold, minRadius=minimum, maxRadius=maximum)
        if circles is not None:
            candidates.extend(circles[0])
    selected = []
    hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
    edges = cv.Canny(gray, 60, 120)
    for a, b, r in sorted(candidates, key=lambda c: -c[2]):
        # Merge repeated detections of the same rim or growth ring.
        if any(np.hypot(a-d,b-e) < (0.5*max(r,t) if config.get('wood') else 0.65*min(r,t))
               and (config.get('wood') or max(r,t)/min(r,t) < 2) for d,e,t in selected):
            continue
        if config.get('polygon') and cv.pointPolygonTest(
                np.array(config['polygon'],np.float32), (a/stretch,b), False) < 0:
            continue
        px = hsv[min(round(b),h-1), min(round(a),crop.shape[1]-1)]
        if config.get('wood') and (px[2] < 60 or px[0] > 40):
            continue
        if config.get('exclude_side') and b < 150 and a > 325:
            continue
        if config.get('rim_check'):
            support = 0
            for angle in np.linspace(0, 2*np.pi, 48, endpoint=False):
                xx, yy = round(a+r*np.cos(angle)), round(b+r*np.sin(angle))
                margin = max(2, round(r*0.08))
                if 0 <= xx < crop.shape[1] and 0 <= yy < h:
                    support += np.any(edges[max(0,yy-margin):yy+margin+1,
                                            max(0,xx-margin):xx+margin+1])
            if support/48 < 0.65:
                continue
        if config.get('dark_center') and np.mean(crop[min(round(b),h-1),
                                                    min(round(a),crop.shape[1]-1)]) > 140:
            continue
        selected.append((float(a),float(b),float(r)))
    # Map detections back to original coordinates; stretched circles become ellipses.
    return [((a/stretch+x)/scale, (b+y)/scale, r/stretch/scale, r/scale)
            for a,b,r in selected]


def main():
    root = Path(__file__).resolve().parent
    configs = json.loads((root/'parameters.json').read_text(encoding='utf-8-sig'))
    output = root/'results'
    output.mkdir(exist_ok=True)
    summary, positions = [], []
    for name, config in configs.items():
        image = cv.imdecode(np.fromfile(root/'hw3'/name,np.uint8),cv.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read {name}')
        circles = detect(image,config)
        marked = image.copy()
        for i,(x,y,rx,ry) in enumerate(circles,1):
            cv.ellipse(marked,(round(x),round(y)),(round(rx),round(ry)),
                       0,0,360,(0,255,0),max(1,round(image.shape[1]/750)))
            cv.putText(marked,str(i),(round(x),round(y)),cv.FONT_HERSHEY_SIMPLEX,
                       max(.3,image.shape[1]/2200),(0,0,255),1)
            positions.append([name,i,round(x,2),round(y,2),round(rx,2),round(ry,2)])
        cv.imencode('.jpg',marked)[1].tofile(output/name)
        summary.append([name,len(circles)])
        print(f'{name}: {len(circles)}')
    for filename, header, rows in [('counts.csv',['image','count'],summary),
                                 ('circles.csv',['image','id','x','y','radius_x','radius_y'],positions)]:
        with (output/filename).open('w',newline='',encoding='utf-8') as file:
            writer=csv.writer(file);writer.writerow(header);writer.writerows(rows)
    (output/'counts.txt').write_text('\n'.join(f'{n}: {c}' for n,c in summary)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
