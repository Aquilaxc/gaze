import random
import numpy as np
import torch
import cv2


def random_shift(img):
    """
    Random shift to face and eye positions while Training, in order to improve robustness
    """
    r = random.randint(-20, 20)
    img_array = np.array(img)
    img_array += r
    return torch.from_numpy(img_array)


def get_box(label, shift=True):
    """
    :param label: Facial Landmarks. 0-3: left eye; 4-7: right eye; 8-11: mouth.
    :param shift: random shift added to boxes. True in training process, False in test/inference process
    :return: left eye, right eye, face.  boxes(x1, y1, x2, y2), torch tensor.
    """
    left_eye_x1 = int(label[0])
    left_eye_y1 = int(label[1])
    left_eye_x2 = int(label[2])
    left_eye_y2 = int(label[3])
    right_eye_x1 = int(label[4])
    right_eye_y1 = int(label[5])
    right_eye_x2 = int(label[6])
    right_eye_y2 = int(label[7])
    mouth_x1 = int(label[8])
    mouth_y1 = int(label[9])
    mouth_x2 = int(label[10])
    mouth_y2 = int(label[11])

    left_eye_center_x = (left_eye_x1 + left_eye_x2) / 2
    left_eye_center_y = (left_eye_y1 + left_eye_y2) / 2
    left_size = (left_eye_x2 - left_eye_x1) * 1.7
    right_eye_center_x = (right_eye_x1 + right_eye_x2) / 2
    right_eye_center_y = (right_eye_y1 + right_eye_y2) / 2
    right_size = (right_eye_x2 - right_eye_x1) * 1.7
    eye_center_x = (left_eye_center_x + right_eye_center_x) / 2
    eye_center_y = (left_eye_center_y + right_eye_center_y) / 2
    mouth_center_x = (mouth_x1 + mouth_x2) / 2
    mouth_center_y = (mouth_y1 + mouth_y2) / 2
    face_center_x = round((eye_center_x + mouth_center_x) / 2)
    face_center_y = round((eye_center_y + mouth_center_y) / 2)
    face_size = ((left_size + right_size) / 2) / 0.3

    leftEye = [round(left_eye_center_x - left_size / 2), round(left_eye_center_y - left_size / 2),
               round(left_eye_center_x + left_size / 2), round(left_eye_center_y + left_size / 2),
               ]
    rightEye = [round(right_eye_center_x - right_size / 2), round(right_eye_center_y - right_size / 2),
                round(right_eye_center_x + right_size / 2), round(right_eye_center_y + right_size / 2),
                ]
    face = [round(face_center_x - face_size / 2), round(face_center_y - face_size / 2),
            round(face_center_x + face_size / 2), round(face_center_y + face_size / 2),
            ]

    if shift:
        return random_shift(leftEye), random_shift(rightEye), random_shift(face)
    else:
        return torch.tensor(leftEye, dtype=torch.float32), \
               torch.tensor(rightEye, dtype=torch.float32), \
               torch.tensor(face, dtype=torch.float32)


def crop_img(img, box):
    """
    :param img: raw image
    :param height: x1, y1, x2, y2, i.e. up-left & bottom-right points
    :param width: x1, y1, x2, y2, i.e. up-left & bottom-right points
    :param box: x1, y1, x2, y2, i.e. up-left & bottom-right points
    :return: cropped-out part of image
    """
    img = np.array(img)
    height, width, _ = img.shape
    x1, y1, x2, y2 = box.long()
    x1 = np.clip(x1, 0, width)  # Adjust out-of-bound boxes
    x2 = np.clip(x2, 0, width)
    y1 = np.clip(y1, 0, height)
    y2 = np.clip(y2, 0, height)
    img_out = img[y1:y2, x1:x2]
    if img_out.shape[0] != img_out.shape[1]:    # Add pad to rectangular boxes to prevent distortion
        size = max(img_out.shape[0], img_out.shape[1])
        img_pad = np.zeros((size, size, 3), dtype=np.uint8)  # np.zeros default type: float!!! Will conduct to cv2 failure
        img_pad[:img_out.shape[0], :img_out.shape[1]] = img_out
        img_out = img_pad
    return img_out


def input_preprocess(img, leftEye, rightEye, face):
    leftEye_img = crop_img(img, leftEye)    # Crop out the part only contains left eye
    leftEye_img = cv2.resize(leftEye_img, (112, 112), interpolation=cv2.INTER_NEAREST)
    leftEye_img = leftEye_img / 255  # Normalize to [0, 1]
    leftEye_img = leftEye_img.transpose(2, 0, 1)  # h, w, c -> c, h, w

    rightEye_img = crop_img(img, rightEye)  # Crop out the part only contains right eye
    rightEye_img = cv2.resize(rightEye_img, (112, 112), interpolation=cv2.INTER_NEAREST)
    rightEye_img = cv2.flip(rightEye_img, 1)    # Horizontally flip right eye
    rightEye_img = rightEye_img / 255
    rightEye_img = rightEye_img.transpose(2, 0, 1)

    face_img = crop_img(img, face)  # Crop out the part only contains face
    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_NEAREST)
    face_img = face_img / 255
    face_img = face_img.transpose(2, 0, 1)

    rects = torch.cat((leftEye, rightEye, face), 0).type(torch.FloatTensor)  # Boxes' coordinates of eyes and face

    return torch.from_numpy(leftEye_img).type(torch.FloatTensor), \
           torch.from_numpy(rightEye_img).type(torch.FloatTensor), \
           torch.from_numpy(face_img).type(torch.FloatTensor), \
           rects


def exp_smooth(before, now, alpha=0.9):
    if isinstance(before, list):
        before, now = np.array(before), np.array(now)
        after = alpha * now + (1 - alpha) * before
        after = after.tolist()
    elif isinstance(before, tuple):
        before, now = np.array(before), np.array(now)
        after = alpha * now + (1 - alpha) * before
        after = tuple(after)
    elif isinstance(before, np.ndarray):
        before, now = np.array(before), np.array(now)
        after = alpha * now + (1 - alpha) * before
    else:
        after = alpha * now + (1 - alpha) * before
    return after



def vector_to_yawpitch(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.
    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.
    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    vectors = vectors[np.newaxis, :] if vectors.shape == (3, ) else vectors
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out.squeeze()    # yaw, pitch


def yawpitch_to_vector(yawpitch):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.
    Args:
        yawpitch (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.
    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    yawpitch = yawpitch.unsqueeze(0) if yawpitch.size() == (2, ) else yawpitch
    n = yawpitch.size()[0]
    sin = torch.sin(yawpitch)
    cos = torch.cos(yawpitch)
    out = torch.empty((n, 3), device=yawpitch.device)
    out[:, 0] = torch.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = torch.multiply(cos[:, 0], cos[:, 1])
    # out = out * np.pi / 180
    return out


def angular_error_np(gt, predict, reduction='sum'):
    assert gt.shape[0] == predict.shape[0]
    gt = yawpitch_to_vector(gt) if gt.shape[1] == 2 else gt
    predict = yawpitch_to_vector(predict) if predict.shape[1] == 2 else predict
    losses = 0
    for g, p in zip(gt, predict):
        cosine_sim = np.dot(g, p) / (np.linalg.norm(g) * np.linalg.norm(p))
        loss = 1 - cosine_sim
        losses += loss
    if reduction == 'sum':
        return torch.tensor(losses, requires_grad=True)
    if reduction == 'mean':
        return torch.tensor(losses / gt.shape[0], requires_grad=True)


def angular_error(gt, predict, reduction='sum'):
    assert gt.size()[0] == predict.size()[0]
    gt = yawpitch_to_vector(gt) if gt.size()[1] == 2 else gt
    predict = yawpitch_to_vector(predict) if predict.size()[1] == 2 else predict
    losses = 0
    for g, p in zip(gt, predict):
        cosine_sim = torch.matmul(g, p) / (torch.norm(g) * torch.norm(p))
        loss = 1 - cosine_sim
        losses += loss
    if reduction == 'sum':
        return losses
    if reduction == 'mean':
        return losses / gt.size()[0]


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    # if "state_dict" in pretrained_dict.keys():
    #     pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    # else:
    #     pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=True)
    return model
