import json
import pandas as pd
import boto3
import os
import scipy.stats as st
import numpy as np

reko = boto3.client("rekognition")
PATH = "data/fairface/fairface_images/"
RACE_LIST = ["East Asian", "Middle Eastern", "White","Latino_Hispanic", "Indian", "Southeast Asian", "Black"]

# Creates a dict of bytes from an image which can 
# then be fed into Rekognition
def create_image_dict_from_file(photo):
    with open(photo, 'rb') as image:
        return {'Bytes': image.read()}

# Turns JSON encoded data into Python object
def load_from_json(file):
    with open(file, 'r', encoding='utf-8') as fp:
        return json.load(fp)
    
# Given a JSON file, returns the file transformed 
# into a pandas dataframe
def json_to_df(file):
    result = load_from_json(file)
    df = pd.DataFrame.from_dict(result, "index")
    df.index.name = "file"
    df.reset_index(inplace=True)
    df.sort_values(by=["file"], inplace=True)
    df["file"] = df["file"].str.slice(start=9)
    return df

# Combines data/fairface/fairface_label_val.csv and 
# data/fairface/fairface_label_train.csv into one
# dataframe
def fairface_to_df():
    df = pd.read_csv("data/fairface/fairface_label_val.csv").append(pd.read_csv("data/fairface/fairface_label_train.csv"))
    df.rename(columns={"gender": "Gender"}, inplace=True)
    df.rename(columns={"race": "Race"}, inplace=True)
    df.sort_values(by=["file"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

# Cleans up the results data and returns two dataframes,
# one with labels where rekognition returned results,
# and the where it did not return results
def fairface_results_intersection(labels, results):
    bool_labels_with_results = labels["filename"].isin(results["filename"])
    return labels[bool_labels_with_results], labels[~bool_labels_with_results]

# Creates the results and labels dataframes, returning
# a dataframe of the labels and results as well as a 
# dataframe of the pictures that did not return results
def combine_labels_and_results():
    results_df = json_to_df("data/fairface/fairface.json")
    labels_df = fairface_to_df()
    labels_with_results, labels_without_results = fairface_results_intersection(labels_df, results_df)
    complete_df = pd.DataFrame(data=list(results_df["file"]), columns=["filename"])
    complete_df = complete_df.assign(Race = list(labels_with_results["Race"]),
                                    Label = list(labels_with_results["Label"]),
                                    Prediction = [x["Gender"]["Value"] for x in results_df["details"]],
                                    Confidence = [x["Gender"]["Confidence"] for x in results_df["details"]],
                                    Faces = list(results_df["num_faces"]),
                                    Boundingbox = [x["BoundingBox"] for x in results_df["details"]])
    return complete_df, labels_without_results

# Calculates metrics for given labels and predictions at 
# given confidence threshold, returning a dict of the
# results
def metrics_calculator(label_list, pred_list, conf_list, conf_threshold):
    label_set = sorted(set(label_list))
    gender_dict = dict()

    for gender in label_set:
        tp = 0
        fp = 0
        num_pos = 0
        num_neg = 0
        for i,label in enumerate(label_list):
            pred = pred_list[i]
            conf = conf_list[i]
            bool_conf = conf >= conf_threshold
            bool_gender_match = pred == gender and label == gender
            bool_gender_mismatch = pred == gender and label != gender
            tp += (bool_conf and bool_gender_match)
            fp += (bool_conf and bool_gender_mismatch)
            num_pos += (label == gender)
            num_neg += (label != gender)
        
        metric_dict = dict()
    
        metric_dict["Precision"] = tp / (tp + fp)
        metric_dict['Prec_ci_lower'], metric_dict['Prec_ci_upper'] = wilson_ci(tp, tp+fp)
    
        metric_dict["FDR"] = 1 - metric_dict["Precision"]
        metric_dict['FDR_ci_lower'], metric_dict['FDR_ci_upper'] = wilson_ci(fp, tp+fp)
    
        metric_dict["Recall"] = tp / num_pos
        metric_dict['Reca_ci_lower'], metric_dict['Reca_ci_upper'] = wilson_ci(tp, num_pos)
    
        metric_dict["FPR"] = fp / num_neg
        metric_dict['FPR_ci_lower'], metric_dict['FPR_ci_upper'] = wilson_ci(fp, num_neg)
    
        metric_dict["FNR"] = 1 - metric_dict["Recall"]
        metric_dict['FNR_ci_lower'], metric_dict['FNR_ci_upper'] = wilson_ci(num_pos-tp, num_pos)
    
        metric_dict["TP"] = tp
        metric_dict["FP"] = fp
        metric_dict["TP+FP"] = tp + fp
    
        metric_dict["num_pos"] = num_pos
        metric_dict["num_neg"] = num_neg
    
        gender_dict[gender] = metric_dict
    
    return gender_dict

# Returns a dataframe of metrics for all races
def metrics_dataframe(df, conf_threshold):
    result_dict = dict()
    race_list = list(df["Race"])
    
    result_dict["All"] = metrics_calculator(list(df["Label"]), list(df["Prediction"]), 
                                            list(df["Confidence"]), conf_threshold)
    for race in set(race_list):
        race_df = df[df["Race"] == race]
        label_list = list(race_df["Label"])
        pred_list = list(race_df["Prediction"])
        conf_list = list(race_df["Confidence"])
        
        race_dict = metrics_calculator(label_list, pred_list, conf_list, conf_threshold)
        result_dict[race] = race_dict

    return pd.DataFrame.from_dict(result_dict, orient="index"), result_dict

# Merges dict2 into dict1
def merge(dict1, dict2):
    return (dict1.update(dict2))

# Runs images through Rekognition and adds the image 
# details into the returned dictionary until there are
# num images with only on face in the dictionary
def find_images(images_list, num):
    details_dict = dict()
    for image in images_list:
        face_dict = dict()
        image_bytes = create_image_dict_from_file(PATH + image)
        response = reko.detect_faces(Image=image_bytes, Attributes=["ALL"])
        if len(response["FaceDetails"]) == 0:
            continue
        face_dict["details"] = response["FaceDetails"][0]
        face_dict["num_faces"] = len(response["FaceDetails"])
        details_dict[image] = face_dict
        if len(details_dict) == num:
            break
    return details_dict

# Takes a sample from all input races
def take_samples(df, races=RACE_LIST):
    all_dict = dict()
    for race in races:
        race_df = df[df["Race"] == race]
        
        for gender in ["Female", "Male"]:
            race_gender_df = race_df[race_df["Label"] == gender]
            race_gender_images = list(race_gender_df["filename"])
            race_images_dict = find_images(race_gender_images, 20)
            merge(all_dict, race_images_dict)
    return all_dict

# 
def take_and_combine_samples(df):
    all_details_dict = take_samples(df)
    final_df = pd.DataFrame.from_dict(all_details_dict, "index")
    final_df.index.name = "filename"
    final_df.sort_values(by=["filename"], inplace=True)
    final_df.reset_index(inplace=True)
    return combine_labels_and_results2(final_df, df)

#
def sample_and_combine_labels_and_results():
    labels_df = fairface_to_df()
    labels_df.rename(columns={"race": "Race"}, inplace=True)
    labels_df.rename(columns={"Gender": "Label"}, inplace=True)
    labels_df.rename(columns={"file": "filename"}, inplace=True)
    combined_df, none_df = take_and_combine_samples(labels_df)
    return combined_df

# Combines results dataframe and labels dataframe into 
# the intersection and disjoint, returning both
def combine_labels_and_results2(results_df, labels_df):
    labels_with_results, labels_without_results = fairface_results_intersection(labels_df, results_df)
    complete_df = pd.DataFrame(data=list(results_df["filename"]), columns=["filename"])
    complete_df = complete_df.assign(Race = list(labels_with_results["Race"]),
                                    Label = list(labels_with_results["Label"]),
                                    Prediction = [x["Gender"]["Value"] for x in results_df["details"]],
                                    Confidence = [x["Gender"]["Confidence"] for x in results_df["details"]],
                                    Faces = list(results_df["num_faces"]),
                                    Boundingbox = [x["BoundingBox"] for x in results_df["details"]])
    return complete_df, labels_without_results


def wilson_ci(num_hits, num_total, confidence=0.95):
    z = st.norm.ppf((1+confidence)/2)
    phat = num_hits / num_total
    first_part = 2*num_total*phat + z*z
    second_part = z*np.sqrt(z*z - 1/num_total + 4*num_total*phat*(1-phat) + (4*phat-2))+1
    den = 2*(num_total + z*z)
    lower_bound = max(0,(first_part - second_part) / den) if phat!=0 else 0
    upper_bound = min(1,(first_part + second_part) / den) if phat!=1 else 1
    return lower_bound,upper_bound