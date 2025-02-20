from PIL import Image, ImageDraw
import requests

def string_to_list(query):
    lst = query.split(',')
    return [item.strip() for item in lst]

def image_from_url(url):
    return Image.open(requests.get(url, stream=True).raw)

def image_from_path(image_path):
    return Image.open(image_path)

def download_from_url(url):
    image_path = '/content/img.jpg'
    data = requests.get(url).content
    f = open(image_path,'wb')

    f.write(data)
    f.close()

    return image_path

def get_bboxes(output):
    out_str = output[0]

    # Replace single quotes with double quotes
    str_json = out_str.replace("'", '"')

    # Convert to JSON object
    bboxes = json.loads(str_json)

    return bboxes

def draw_box(image_path, bboxes):
    image = Image.open(image_path)

    for box in bboxes:
      draw = ImageDraw.Draw(image)
      draw.rectangle(box['bbox'], outline="red", width=2)
    return image

def image_640(img_path, new_width=640):
    # Open the image
    img = Image.open(img_path)
    
    # Get original dimensions
    original_width, original_height = img.size
    if original_width > 640:
        # Calculate new height while maintaining aspect ratio
        new_height = int((new_width / original_width) * original_height)
        
        # Resize the image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save the resized image
        img_resized.save(img_path)

def calculate_bbox_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    Each box should be in format [x1, y1, x2, y2]
    """
    # Get the coordinates of intersecting rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union

def calculate_dict_iou(truth_dict, predict_dict):
    """
    Calculate IoU for each name_key between truth and predict dictionaries
    Returns a dictionary with average IoU for each key and matching details
    """
    results = {}
    
    for name_key in truth_dict:
        if name_key not in predict_dict:
            results[name_key] = {
                'average_iou': 0,
                'matched_pairs': [],
                'unmatched_truth': len(truth_dict[name_key]),
                'unmatched_pred': 0
            }
            continue
            
        truth_boxes = truth_dict[name_key]
        pred_boxes = predict_dict[name_key]
        
        # Calculate IoU matrix between all truth and prediction boxes
        iou_matrix = []
        for truth_box in truth_boxes:
            row = []
            for pred_box in pred_boxes:
                iou = calculate_bbox_iou(truth_box, pred_box)
                row.append(iou)
            iou_matrix.append(row)
        
        # Match boxes using greedy approach
        matched_pairs = []
        used_truth = set()
        used_pred = set()
        
        while True:
            max_iou = 0
            max_truth_idx = -1
            max_pred_idx = -1
            
            # Find the highest IoU among remaining boxes
            for i in range(len(truth_boxes)):
                if i in used_truth:
                    continue
                for j in range(len(pred_boxes)):
                    if j in used_pred:
                        continue
                    if iou_matrix[i][j] > max_iou:
                        max_iou = iou_matrix[i][j]
                        max_truth_idx = i
                        max_pred_idx = j
            
            # If no more matches found or IoU too low, break
            if max_iou < 0.1:  # You can adjust this threshold
                break
                
            matched_pairs.append({
                'truth_idx': max_truth_idx,
                'pred_idx': max_pred_idx,
                'iou': max_iou
            })
            used_truth.add(max_truth_idx)
            used_pred.add(max_pred_idx)
        
        # Calculate average IoU and store results
        average_iou = sum(pair['iou'] for pair in matched_pairs) / len(matched_pairs) if matched_pairs else 0
        
        results[name_key] = {
            'average_iou': average_iou,
            'matched_pairs': matched_pairs,
            'unmatched_truth': len(truth_boxes) - len(matched_pairs),
            'unmatched_pred': len(pred_boxes) - len(matched_pairs)
        }
    
    # Handle keys in predict_dict that are not in truth_dict
    for name_key in predict_dict:
        if name_key not in truth_dict:
            results[name_key] = {
                'average_iou': 0,
                'matched_pairs': [],
                'unmatched_truth': 0,
                'unmatched_pred': len(predict_dict[name_key])
            }
    
    return results

def calculate_accuracy_metrics(truth_dict, predict_dict, iou_threshold=0.5):
    """
    Calculate accuracy metrics based on IoU results
    Returns precision, recall, F1-score, and mean IoU for each class and overall
    """
    results = calculate_dict_iou(truth_dict, predict_dict)
    metrics = {
        'per_class': {},
        'overall': {}
    }
    
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_iou_sum = 0
    total_matches = 0
    
    # Calculate per-class metrics
    for class_name, class_results in results.items():
        # Count matches above IoU threshold as true positives
        true_positives = sum(1 for pair in class_results['matched_pairs'] 
                           if pair['iou'] >= iou_threshold)
        
        # False positives are unmatched predictions plus matches below threshold
        false_positives = (class_results['unmatched_pred'] + 
                         sum(1 for pair in class_results['matched_pairs'] 
                             if pair['iou'] < iou_threshold))
        
        # False negatives are unmatched ground truths plus matches below threshold
        false_negatives = (class_results['unmatched_truth'] + 
                         sum(1 for pair in class_results['matched_pairs'] 
                             if pair['iou'] < iou_threshold))
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = class_results['average_iou']
        
        metrics['per_class'][class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_iou': mean_iou,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
        # Accumulate totals for overall metrics
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        total_iou_sum += mean_iou * len(class_results['matched_pairs'])
        total_matches += len(class_results['matched_pairs'])
    
    # Calculate overall metrics
    overall_precision = (total_true_positives / (total_true_positives + total_false_positives) 
                        if (total_true_positives + total_false_positives) > 0 else 0)
    overall_recall = (total_true_positives / (total_true_positives + total_false_negatives)
                     if (total_true_positives + total_false_negatives) > 0 else 0)
    overall_f1 = (2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
                 if (overall_precision + overall_recall) > 0 else 0)
    overall_miou = total_iou_sum / total_matches if total_matches > 0 else 0
    
    metrics['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'mean_iou': overall_miou,
        'true_positives': total_true_positives,
        'false_positives': total_false_positives,
        'false_negatives': total_false_negatives
    }
    
    return metrics