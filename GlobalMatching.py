import cv2
import numpy as np
from typing import List,Tuple

def GlobalMatching(frame: cv2.typing.MatLike, 
                   template: cv2.typing.MatLike, 
                   template_name: str, 
                   search_area: List[Tuple[int, int]],
                   mask: cv2.typing.MatLike = None) -> Tuple[bool,List[int], str, float]:
    if mask is not None:
        if not isinstance(mask,cv2.typing.MatLike):
            raise ValueError ("Mask type incorrect")
        if mask.shape != template.shape:
            raise ValueError ("Mask size do not match template size")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        threshold = 0.95

        cutted_frame = None

        if search_area != [(0,0), (0,0)]:
            top_left_search = search_area[0]
            bottom_right_search = search_area[1]
            cutted_frame = frame[top_left_search[1]:bottom_right_search[1],top_left_search[0]:bottom_right_search[0]]
            if cutted_frame.shape[0]<template.shape[0] or cutted_frame.shape[1]<template.shape[1]:
                raise ValueError(f"Searching is too small with respect to the template:\nCutted_frame = Height {cutted_frame.shape[0]} Width {cutted_frame.shape[1]}, template = height {template.shape[0]} width {template.shape[1]}")
            res = cv2.matchTemplate(cutted_frame, template, cv2.TM_CCORR_NORMED, mask = mask)
        else:
            res = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED, mask = mask)

        # locations = np.where (res>= threshold)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val>threshold:
            bottom_left = max_loc
            if cutted_frame.any():
                bottom_left = [bottom_left[0] + top_left_search[0], bottom_left[1] + bottom_right_search[1]]
            
            return (True, max_loc, template_name, max_val)

        else:
            return (False, [0, 0], template_name, max_val)
    except Exception as e:
        print(e)
        return(False,[0, 0], template_name, 0)