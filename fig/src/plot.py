# %% [markdown]
# # map + detection count plots

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dual_plot(data_list, yrange):
    """
    Create a dual subplot with mAP values on the left and detection counts on the right.
    
    Parameters:
    data_list: List of dictionaries containing x, y, and label data
               Expected format:
               [
                   {"x": [...], "y": [...], "label": "mAP Value"},
                   {"x": [...], "y": [...], "label": "Near-OoD"},
                   {"x": [...], "y": [...], "label": "Far-OoD"}
               ]
    """
    
    # Extract data based on labels
    map_data = None
    near_ood_data = None
    far_ood_data = None
    
    for data in data_list:
        if data["label"] == "mAP Value":
            map_data = data
        elif data["label"] == "Near-OoD":
            near_ood_data = data
        elif data["label"] == "Far-OoD":
            far_ood_data = data
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        horizontal_spacing=0.12
    )
    
    # Left plot: mAP Value
    if map_data:
        fig.add_trace(
            go.Scatter(
                x=map_data["x"], 
                y=map_data["y"],
                mode='lines+markers',
                name='mAP',
                line=dict(color='orange', width=4),
                marker=dict(size=[12 if i%2==0 else 0 for i in range(len(map_data["x"]))], color='orange'),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Right plot: Detection Count
    if near_ood_data:
        fig.add_trace(
            go.Scatter(
                x=near_ood_data["x"], 
                y=near_ood_data["y"],
                mode='lines+markers',
                name='Near-OoD',
                line=dict(color='red', width=4),
                marker=dict(size=12, color='red'),
            ),
            row=1, col=2
        )
    
    if far_ood_data:
        fig.add_trace(
            go.Scatter(
                x=far_ood_data["x"], 
                y=far_ood_data["y"],
                mode='lines+markers',
                name='Far-OoD',
                line=dict(color='blue', width=4),
                marker=dict(size=12, color='blue'),
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        font=dict(
            family="Arial",
            size=24,  # Larger default font size for papers
        ),
        legend=dict(
            font=dict(size=20),  # Legend font size
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top'
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        width=1200,
        height=600,
        showlegend=True,
        template="plotly"
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text="Epoch",
        dtick=2,
        tick0=0
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="mAP Value",
        row=1, col=1,
        range=[0, 0.4],
        dtick=0.05
    )
    
    fig.update_yaxes(
        title_text="Detection Count",
        row=1, col=2,
        range=yrange,
        dtick=200
    )
    return fig


# yolo finetune trend
# Example usage with your data (only even x values: 0,2,4,...,20):
data_list = [
    {
        "x": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "y": [0.315, 0.302, 0.268, 0.271, 0.285, 0.298, 0.3, 0.302, 0.304, 0.305, 0.306],
        "label": "mAP Value"
    },
    {
        "x": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "y": [708, 160, 220, 170, 160, 170, 160, 150, 140, 140, 145],
        "label": "Near-OoD"
    },
    {
        "x": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "y": [641, 130, 240, 140, 240, 150, 140, 130, 130, 140, 136],
        "label": "Far-OoD"
    }
]

# Create and show the plot
fig = create_dual_plot(data_list, [100, 800])
fig.write_image("/Users/hugo/67cee99472a50e09554ac32f/fig/finetune_trend.png", width=1200, height=600, scale=2,engine="kaleido")

# %%
# faster-rcnn finetune trend
epochs = [i for i in range(0, 21, 2)]  # [0, 2, 4, ..., 20]

# 左图数据：mAP Value
map_values = [0.298, 0.298, 0.298, 0.298, 0.298, 0.298, 0.298, 0.298, 0.298, 0.298, 0.295]

# 右图数据：Detection Count
near_ood_detections = [1080, 400, 400, 390, 415, 420, 440, 430, 400, 395, 405]
far_ood_detections = [880, 265, 255, 270, 290, 285, 320, 345, 310, 310, 325]

# Convert to data_list format
data_list = [
    {
        "x": epochs,
        "y": map_values,
        "label": "mAP Value"
    },
    {
        "x": epochs,
        "y": near_ood_detections,
        "label": "Near-OoD"
    },
    {
        "x": epochs,
        "y": far_ood_detections,
        "label": "Far-OoD"
    }
]

# Now you can use it with the function
fig = create_dual_plot(data_list, [200, 1100])
fig.write_image("/Users/hugo/67cee99472a50e09554ac32f/fig/fasterrcnn_trend.png", width=1200, height=600, scale=2,engine="kaleido")

# %% [markdown]
# # confidence and detection count plots

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 数据
epochs = [0, 1, 2, 3, 4]

# 左图数据：Average Confidence
far_ood_confidence = [0.45923389114898905, 0.29833004202184343, 0.2982504248778878, 0.2903130332122859, 0.29386594839613817, 0.28663075617427164]
# [0.4592, 0.438, 0.428, 0.407, 0.436]
near_ood_confidence = [0.4558119750680478, 0.30615549992975455, 0.3030317713634297, 0.296946633359499, 0.298438067904962, 0.2933210852796994]
# [0.4558, 0.413, 0.406, 0.374, 0.397]

# 右图数据：Total Detections
far_ood_detections = [641, 194, 173, 151, 171]
near_ood_detections = [708, 247, 212, 185, 205]

# 创建子图（移除标题）
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    horizontal_spacing=0.1
)

# 左图：Average Confidence
fig.add_trace(
    go.Scatter(
        x=epochs, 
        y=near_ood_confidence,
        mode='lines+markers',
        name='Near-OOD',
        line=dict(color='red', width=4),
        marker=dict(size=12, color='red')
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=epochs, 
        y=far_ood_confidence,
        mode='lines+markers',
        name='Far-OOD',
        line=dict(color='blue', width=4),
        marker=dict(size=12, color='blue')
    ),
    row=1, col=1
)



# 右图：Total Detections
fig.add_trace(
    go.Scatter(
        x=epochs, 
        y=near_ood_detections,
        mode='lines+markers',
        name='Near-OOD',
        line=dict(color='red', width=4),
        marker=dict(size=12, color='red'),
        showlegend=False
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=epochs, 
        y=far_ood_detections,
        mode='lines+markers',
        name='Far-OOD',
        line=dict(color='blue', width=4),
        marker=dict(size=12, color='blue'),
        showlegend=False
    ),
    row=1, col=2
)



# 更新布局 - 将图例移到右图的右上角
fig.update_layout(
    font=dict(
        family="Arial",
        size=24,  # Larger default font size for papers
    ),
    legend=dict(
        font=dict(size=20),  # Legend font size
        x=0.98,
        y=0.98,
        xanchor='right',
        yanchor='top'
    ),
    margin=dict(l=20, r=20, t=20, b=20)
)

# 更新x轴（设置为整数）
fig.update_xaxes(
    title_text="Epoch",
    dtick=1,
    tick0=0
)

# 更新y轴
fig.update_yaxes(
    title_text="Average Confidence",
    row=1, col=1,
    range=[0.2, 0.5],
    dtick=0.05
)

fig.update_yaxes(
    title_text="Detection Count",
    row=1, col=2,
    range=[100, 800],
    dtick=100
)
fig.update_layout(template="plotly")
fig.write_image("/Users/hugo/67cee99472a50e09554ac32f/fig/confidence_epoch_trends.png", width=1200, height=600, scale=2,engine="kaleido")

# %% [markdown]
# # few shot on horse

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np  # For potential error handling

def create_solo_plot(data, data2, x_name, y_name, yrange):
    try:
        # Convert x values to numbers for both data and data2
        x_num_data = [float(x) for x in data["x"]]  # e.g., ["0", "5"] -> [0.0, 5.0]
        x_num_data2 = [float(x) for x in data2["x"]]  # Assuming data2 has the same x structure
        
        # Ensure x values are the same for both datasets (for tick alignment)
        if set(x_num_data) != set(x_num_data2):
            raise ValueError("x values in data and data2 must be identical for consistent ticks.")
        
        # Create subplots
        fig = make_subplots(rows=1, cols=1)
        
        # Add first trace
        fig.add_trace(
            go.Scatter(
                x=x_num_data,  # Use numerical x values
                y=data["y"],
                mode='lines+markers',
                name=data["name"],
                line=dict(color='red', width=4),
                marker=dict(size=12, color='red'),
            ),
            row=1, col=1
        )
        
        # Add second trace
        fig.add_trace(
            go.Scatter(
                x=x_num_data2,  # Use numerical x values
                y=data2["y"],
                mode='lines+markers',
                name=data2["name"],
                line=dict(color='blue', width=4),
                marker=dict(size=12, color='blue'),
            ),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            font=dict(
                family="Arial",
                size=24,  # Larger default font size for papers
            ),
            legend=dict(
                font=dict(size=20),  # Legend font size
                x=0.98,
                y=0.98,
                xanchor='right',
                yanchor='top',
            ),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        # Update x-axes: Set ticks to match the number of x values
        fig.update_xaxes(
            title_text=x_name,
            tickvals=x_num_data,  # Ticks will be at each x value (e.g., [0, 5, 10, 15, 20])
        )
        
        # Update y-axes
        fig.update_yaxes(
            title_text=y_name,
            row=1, col=1,
            range=yrange,
        )
        fig.update_layout(template="plotly")
        return fig
    
    except ValueError as e:
        print(f"Error: {e}. Ensure all x values are numeric strings.")
        return None  # Or handle as needed

# Example usage with your data
data = {"x": ["0", "5", "10", "15", "20"],
        "y": [140, 120, 80, 60, 50],
        "name": "Train"}

data2 = {"x": ["0", "5", "10", "15", "20"],
         "y": [1400, 1150, 640, 500, 460],
         "name": "Val"}
# Create and show the plot
fig = create_solo_plot(data, data2, "Epoch", "Detection Count", [0, 1500])

if fig:
    fig.write_image("/Users/hugo/67cee99472a50e09554ac32f/fig/few_shot_horse_trends.png", width=600, height=600, scale=2,engine="kaleido")


# %% [markdown]
# # few shot on sample sizes

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

data = {
  "percent_0": {
    "total": 2566,
    "pedestrian": 2045
  },
  "percent_5": {
    "total": 1702,
    "pedestrian": 1256
  },
  "percent_10": {
    "total": 917,
    "pedestrian": 633
  },
  "percent_15": {
    "total": 588,
    "pedestrian": 363
  },
  "percent_20": {
    "total": 475,
    "pedestrian": 281
  }
}

import json
import plotly.graph_objects as go

# Convert percentage keys to actual few-shot sample counts
# Formula: percentage * 1277
few_shot_samples = []
total_detections = []
pedestrian_detections = []

for key, values in data.items():
    # Extract percentage from key (e.g., "percent_5" -> 5)
    percentage = int(key.split('_')[1])
    # Calculate actual few-shot samples
    samples = int(percentage * 1277 / 100)
    
    few_shot_samples.append(samples)
    total_detections.append(values['total'])
    pedestrian_detections.append(values['pedestrian'])

# Create the plot following the style from your example
fig = go.Figure()

# Add total detections line (similar to Near-OoD in red)
fig.add_trace(go.Scatter(
    x=few_shot_samples, 
    y=total_detections, 
    mode='lines+markers', 
    name="Total Count",
    line=dict(color='red', width=4),
    marker=dict(size=12, color='red')
))

# Add pedestrian detections line (similar to Far-OoD in blue)
fig.add_trace(go.Scatter(
    x=few_shot_samples, 
    y=pedestrian_detections, 
    mode='lines+markers', 
    name="Pedestrian Count",
    line=dict(color='blue', width=4),
    marker=dict(size=12, color='blue')
))


fig.update_layout(
    font=dict(
        family="Arial",
        size=24,  # Larger default font size for papers
    ),
    legend=dict(
        font=dict(size=20),  # Legend font size
        x=0.98,
        y=0.98,
        xanchor="right",
        yanchor="top"
    ),
    height=600, 
    width=600,
    showlegend=True,
    margin=dict(l=20, r=20, t=20, b=20)
)
fig.update_xaxes(title_text="Few-Shot Samples")
fig.update_yaxes(title_text="Detection Count")
fig.update_layout(template="plotly")
# Show plot
fig.write_image("/Users/hugo/67cee99472a50e09554ac32f/fig/fsl_ablation_sample_size.png", width=600, height=600, scale=2,engine="kaleido")

# %% [markdown]
# # few shot ablation on categories

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_multi_animal_plot(all_data, x_name, y_name):
    try:
        fig = make_subplots(
            rows=1, cols=len(all_data),
            subplot_titles=[name for name, _ in all_data],
            shared_xaxes=True,
            shared_yaxes=False,
            horizontal_spacing=0.1
        )
        
        colors = {'train': 'red', 'val': 'blue'}

        for col_idx, (animal_name, data) in enumerate(all_data, 1):
            try:
                x_train = [float(str(x).lower().replace('epoch', '')) if isinstance(x, (str, int, float)) else float(x) 
                           for x in data["Train"]["x"]]
                x_val = [float(str(x).lower().replace('epoch', '')) if isinstance(x, (str, int, float)) else float(x) 
                         for x in data["Val"]["x"]]
            except ValueError as e:
                raise ValueError(f"Could not convert x values to float for {animal_name}: {e}. Ensure x values are numeric or 'epochN' format.")

            if set(x_train) != set(x_val):
                raise ValueError(f"x values for {animal_name} Train and Val must match after conversion.")
            
            fig.add_trace(
                go.Scatter(
                    x=x_train, y=data["Train"]["y"],
                    mode='lines+markers', name='Train',
                    line=dict(color=colors['train'], width=4),
                    marker=dict(size=12, color=colors['train']),
                    showlegend=(col_idx == len(all_data)),
                    legendgroup='trainval'
                ),
                row=1, col=col_idx
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_val, y=data["Val"]["y"],
                    mode='lines+markers', name='Val',
                    line=dict(color=colors['val'], width=4),
                    marker=dict(size=12, color=colors['val']),
                    showlegend=(col_idx == len(all_data)),
                    legendgroup='trainval'
                ),
                row=1, col=col_idx
            )
            
            fig.update_xaxes(
                title_text=x_name if col_idx == (len(all_data) + 1) // 2 else "",
                row=1, col=col_idx
            )
            
            fig.update_yaxes(
                title_text=y_name if col_idx == 1 else "",
                row=1, col=col_idx
            )
        
        # Update layout to position the legend inside the plot area
        fig.update_layout(
            font=dict(
                family="Arial",
                size=24,  # Larger default font size for papers
            ),
            legend=dict(
                font=dict(size=20),  # Legend font size
                x=0.98,
                y=0.98,
                xanchor='right',
                yanchor='top',
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # After creating the multi-animal subplot (with subplot_titles), update the font size of the subplot titles to 22 to match axis titles.
        # Example:
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=22)
        
        return fig
    
    except Exception as e:
        print(f"Error in create_multi_animal_plot: {e}")
        return None

data_horse = {
    "Train": {"x": ["epoch0", "epoch5", "epoch10", "epoch15", "epoch20"], "y": [118, 80, 50, 40, 30]},
    "Val": {"x": ["epoch0", "epoch5", "epoch10", "epoch15", "epoch20"], "y": [1150, 920, 450, 350, 310]}
}
data_monkey = {
    "Train": {"x": ["epoch0", "epoch5", "epoch10", "epoch15", "epoch20"], "y": [75, 25, 15, 10, 8]},
    "Val": {"x": ["epoch0", "epoch5", "epoch10", "epoch15", "epoch20"], "y": [600, 230, 80, 50, 40]}
}
data_cattle = {
    "Train": {"x": ["epoch0", "epoch5", "epoch10", "epoch15", "epoch20"], "y": [220, 80, 40, 20, 15]},
    "Val": {"x": ["epoch0", "epoch5", "epoch10", "epoch15", "epoch20"], "y": [2150, 1350, 600, 380, 350]}
}

all_data = [("Horse", data_horse), ("Monkey", data_monkey), ("Cattle", data_cattle)]
fig_multi = create_multi_animal_plot(all_data, "Epoch", "Detection Count")
if fig_multi:
    fig_multi.write_image("/Users/hugo/67cee99472a50e09554ac32f/fig/fsl_ablation_categories.png", width=1800, height=600, scale=2,engine="kaleido")

# %%



