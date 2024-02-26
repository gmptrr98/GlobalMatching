import cv2
import numpy as np
from typing import List,Tuple

def GlobalMatching(frame: cv2.typing.MatLike, 
                   template: cv2.typing.MatLike, 
                   template_name: str, 
                   search_area: List[Tuple[int, int]],
                   mask: cv2.typing.MatLike = None) -> Tuple[bool,List[int], str, float]: 
    
    """_summary_

    Args:
        frame (cv2.typing.MatLike): Matlike image.
        template (cv2.typing.MatLike): Matlike image.
        template_name (str): Name of template search request, used just to identify return value.
        search_area (List[Tuple[int, int]]): (Top Left Corner [x1 y1], Bottom Rigth Corner [x2 y2]).
        mask (cv2.typing.MatLike, optional): Matlike image used as a mask on template. Defaults to None.

    Raises:
        ValueError incorrect mask type: Mask is not passed as a Matlike.
        ValueError incorrect mask shape: Mask shape is not the same as template. 
        ValueError incoherent search area: Search area is smaller than template.

    Returns:
        Tuple[bool,List[int], str, float]: (Presence of template in frame, Bottom left corner of template in frame [x y], Template name, Confidence level)
    """    
    
    if mask is not None:
        if not isinstance(mask,cv2.typing.MatLike):
            raise ValueError ("Mask type incorrect")
        if mask.shape != template.shape:
            raise ValueError ("Mask size do not match template size")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    try:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        threshold = 0.9

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
                bottom_left = [bottom_left[0] + top_left_search[0], bottom_left[1] + top_left_search[1]]
            
            return (True, bottom_left, template_name, max_val)

        else:
            return (False, [0, 0], template_name, max_val)
    except Exception as e:
        print(e)
        return(False,[0, 0], template_name, 0)