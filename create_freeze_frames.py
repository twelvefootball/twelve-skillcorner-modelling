# -*- coding: utf-8 -*-
"""
Created on Fri May  9 09:39:47 2025

Code to generate freeze frames from the start and end frames 
of the events in the dynamic events

@author: gustimorth
"""
import json
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get data_path from environment variable
data_path = os.getenv("DATA_DIR", "")

#Set the data path to the JSONL file. Needs to be specified from the root of all repositories
df_matches = pd.read_parquet(f"{data_path}/matches.parquet")

if not os.path.exists(f"{data_path}/freeze"):
    os.makedirs(f"{data_path}/freeze")

#match_id = df_matches['id'].values[0]
for match_id in df_matches['id'].values:
    try:
        df_events = pd.read_parquet(f"{data_path}/dynamic/{match_id}.parquet")
    except:
        print(f"No dynamic events for match {match_id}")
        continue
    
    # Extract all end and start frames from the events
    frames = list(set(df_events['frame_start'].to_list() + df_events['frame_end'].to_list()))
    
    with open(f"{data_path}/tracking/{match_id}.json", 'r') as file:
        tracking_data = json.load(file)
    #
    #tracking_demo = tracking_data[28000:29000]
    # Example of how to transform the data into a data frame for easier manipulation
    # Reading tracking data
    custom_data = []
    for d in tracking_data:
        #Extract the relevant frames
        if d.get('frame') in frames and d.get('timestamp') is not None:
            for p in d['player_data']:
                custom_data.append({
                    'time': d.get('timestamp'),
                    'frame': int(d.get('frame')),
                    'period': d.get('period'),
                    'player_id': p.get("player_id"),
                    'is_detected':p.get('is_detected'),
                    'is_ball':False,
                    'x': p.get('x'),
                    'y': p.get('y'),
                    'visible_area':d.get('image_corners_projection')
                })
            #Add the ball
            ball = d.get('ball_data')
            if ball['is_detected'] is not None:
                custom_data.append({
                    'time': d.get('timestamp'),
                    'frame': int(d.get('frame')),
                    'period': d.get('period'),
                    'player_id': -1,
                    'is_detected':ball.get('is_detected'),
                    'is_ball':True,
                    'x': ball.get('x'),
                    'y': ball.get('y'),
                    'visible_area':d.get('image_corners_projection')
                    }
                )
        
    
    # Transform to dataframe
    df_frames = pd.DataFrame(custom_data)
    print(f"Generated freeze frames for match {match_id}")
    
    # Save as freeze frames 
    df_frames.to_parquet(f"{data_path}/freeze/{match_id}.parquet")

