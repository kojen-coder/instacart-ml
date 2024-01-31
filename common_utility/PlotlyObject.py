import pandas as pd
import plotly.graph_objs as go


def create_table_trace(df, visible=False):
    trace = go.Table(
        header=dict(values=list(df.columns),
                fill={"color":'royalblue'},
                font=dict(color='white', size=12),
                align='left'),
        cells=dict(values=[df[name] for name in df.columns],
               font=dict(size=12),
             fill={"color":'lavender'},
               align='left'), visible=visible)
    return trace

def visible_true_false_list(list_length, group_length):
    '''
    create visible list for plotly update menu
    :param list_length: total number of pltoly object in one graph
    :param group_length: how many element in each group
    :return: list of list
    '''
    group_num = int(list_length / group_length)
    result_list = []
    for i in range(group_num):
        temp_list = [False] * i * group_length + group_length * [True] + (
            list_length - (i + 1) * group_length) * [
            False]
        result_list.append(temp_list)
    return result_list


def buttons_generator(label_list, visible_list):
    buttons_list = []
    for i, label in enumerate(label_list):
        buttons_list.append(dict(label=label, method="update", args=[{"visible": visible_list[i]}]))
    return buttons_list

def buttons_generator_annotations_title(
    label_list, visible_list, annotation_list,
    title_list
):
    buttons_list = []
    for i, label in enumerate(label_list):
        buttons_list.append(
            dict(label=label, method="update", 
                 args=[
                     {"visible": visible_list[i]},
                     {
                         "annotations": annotation_list[i],
                         "title": title_list[i],
                     }
                 ]))
    return buttons_list

def create_scatter_trace(df, col_x, col_y, name, visible=False, mode='lines'):
    return go.Scatter(x=df[col_x],
                      y=df[col_y],
                      name=name,
                      mode=mode,
                      visible=visible)

def create_bar(df, x_col, y_col, name, visible=False):
    return go.Bar(x=df[x_col], y=df[y_col],
           name=name, visible=visible)

def generate_text(df, input_col):
    df['ratio'] = (df[input_col] / df[input_col].sum()).round(2)
    df['text'] = df['ratio'].apply(lambda x: f'{x:.2%}')
    df['text'] = df[input_col].astype(str) + " (" + df['text'] + ")"
    return df

def buttons_generator(label_list, visible_list):
    buttons_list = []
    for i, label in enumerate(label_list):
        buttons_list.append(dict(label=label, method="update", args=[{"visible": visible_list[i]}]))
    return buttons_list
