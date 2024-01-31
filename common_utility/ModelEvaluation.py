from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, mean_absolute_error, r2_score, mean_squared_error
from plotly.graph_objs import *
from plotly.offline import iplot
from math import sqrt
from sklearn.linear_model import LinearRegression
import plotly
from common_utility.PlotlyObject import visible_true_false_list

import plotly.graph_objs as go
import pandas as pd
import numpy as np


def create_cohort_summary(df, cohort_type, timestamp,  pred_type, label=None, round=2):
    df_size = df.shape[0]
    df_time_max = df[timestamp].dt.date.max()
    df_time_min = df[timestamp].dt.date.min()
    if pred_type == 'classification':
        df_pos = df[df[label] == 1].shape[0] / df_size
        result = pd.DataFrame({"cohort_type": [cohort_type], "size": [df_size],
                      "postive_rate": [df_pos], "start_time": [df_time_min],
                               "end_time": [df_time_max]}).round(round)
    elif pred_type == 'regression':
        result = pd.DataFrame({"cohort_type": [cohort_type], "size": [df_size], "start_time": [df_time_min],
                               "end_time": [df_time_max]})
    return result

def generate_confusion_matrix_heat_map_custom(df, title="", label_col='label', pred_col='pred'):
    '''
    :param df: final dataframe which has the true label and predictions column
    :param thershold: Set Threshold
    :param label_col: True label column in the dataframe
    :param pred_col: Prediction column in the dataframe
    :return: Confusion Matrix Heat map 
    '''
    TN, FP, FN, TP = confusion_matrix(df[label_col], df[pred_col]).ravel()
    
    labelsy = ['True', 'False']
    labelsx = ['False', 'True']
    trace1 = {
            "x": ["Positive", "Negative"],
            "y": ["Negative", "Positive"],
            "z": [[FN, TN], [TP, FP]],
            "autocolorscale": False,
            "colorscale": "Reds",
            "type": "heatmap"
        }
    data = [trace1]
    layout = {
        "barmode": "overlay",
        "height": 600,
        "title": title,
        "width": 600,
        "xaxis": {
            "title": "<b>Confirmed Positive Patients",
            "titlefont": {
                "color": "#7f7f7f",
                "size": 14
            }
        },
        "yaxis": {
            "title": "<b>Predicted Positive Patients",
            "tickangle": -90,
            "titlefont": {
                "color": "#7f7f7f",
                "size": 14
            }
        },
        "annotations": [
            {
                "xref": "x1",
                "yref": "y1",
                "text": "<b>False Negative:" + str(FN),
                "y": "Negative",
                "x": "Positive",
                "font": {
                    "color": "black"
                },
                "showarrow": False
            },
            {
                "xref": "x1",
                "yref": "y1",
                "text": "<b>True Positive:" + str(TP),
                "y": "Positive",
                "x": "Positive",
                "font": {
                    "color": "black"
                },
                "showarrow": False
            },
            {
                "xref": "x1",
                "yref": "y1",
                "text": "<b>True Negative:" + str(TN),
                "y": "Negative",
                "x": "Negative",
                "font": {
                    "color": "black"
                },
                "showarrow": False
            },
            {
                "xref": "x1",
                "yref": "y1",
                "text": "<b>False Positive:" + str(FP),
                "y": "Positive",
                "x": "Negative",
                "font": {
                    "color": "black"
                },
                "showarrow": False
            }
        ]
    }
    fig = go.Figure(data=data, layout=layout)
    return (fig)
    
def generate_confusion_matrix_heat_map(df, threshold=0.5, label_col='label', pred_col='pred'):
    '''
    :param df: final dataframe which has the true label and predictions column
    :param thershold: Set Threshold
    :param label_col: True label column in the dataframe
    :param pred_col: Prediction column in the dataframe
    :return: Confusion Matrix Heat map 
    '''
    TN, FP, FN, TP = confusion_matrix(df[label_col], df[pred_col]).ravel()
    
    labelsy = ['True', 'False']
    labelsx = ['False', 'True']
    trace1 = {
            "x": ["Positive", "Negative"],
            "y": ["Negative", "Positive"],
            "z": [[FN, TN], [TP, FP]],
            "autocolorscale": False,
            "colorscale": "Reds",
            "type": "heatmap"
        }
    data = [trace1]
    layout = {
        "barmode": "overlay",
        "height": 600,
        "title": "<b>Confusion Matrix (Threshold:"+str(threshold)+")",
        "width": 600,
        "xaxis": {
            "title": "<b>Confirmed Positive Patients",
            "titlefont": {
                "color": "#7f7f7f",
                "size": 14
            }
        },
        "yaxis": {
            "title": "<b>Predicted Positive Patients",
            "tickangle": -90,
            "titlefont": {
                "color": "#7f7f7f",
                "size": 14
            }
        },
        "annotations": [
            {
                "xref": "x1",
                "yref": "y1",
                "text": "<b>False Negative:" + str(FN),
                "y": "Negative",
                "x": "Positive",
                "font": {
                    "color": "black"
                },
                "showarrow": False
            },
            {
                "xref": "x1",
                "yref": "y1",
                "text": "<b>True Positive:" + str(TP),
                "y": "Positive",
                "x": "Positive",
                "font": {
                    "color": "black"
                },
                "showarrow": False
            },
            {
                "xref": "x1",
                "yref": "y1",
                "text": "<b>True Negative:" + str(TN),
                "y": "Negative",
                "x": "Negative",
                "font": {
                    "color": "black"
                },
                "showarrow": False
            },
            {
                "xref": "x1",
                "yref": "y1",
                "text": "<b>False Positive:" + str(FP),
                "y": "Positive",
                "x": "Negative",
                "font": {
                    "color": "black"
                },
                "showarrow": False
            }
        ]
    }
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig, filename='datetime-heatmap')
    

    
    
def confusion_matrix_trace_and_dict(
    df, label_col, pred_col,
    legend_name = '', visible = True
):
    TN, FP, FN, TP = confusion_matrix(df[label_col], df[pred_col]).ravel()
    confusion_matrix_dict = {}
    confusion_matrix_dict['TN'] = TN
    confusion_matrix_dict['FP'] = FP
    confusion_matrix_dict['FN'] = FN
    confusion_matrix_dict['TP'] = TP
    trace1 = {
        "x": ["Positive", "Negative"],
        "y": ["Negative", "Positive"],
        "z": [[FN, TN], [TP, FP]],
        'name': legend_name,
        'visible': visible,
        "autocolorscale": False,
        "colorscale": "Reds",
        "type": "heatmap"
        }
    annotations = [
        {
            "xref": "x1",
            "yref": "y1",
            "text": "<b>False Negative:" + str(FN),
            "y": "Negative",
            "x": "Positive",
            "font": {
                "color": "black"
            },
            "showarrow": False
        },
        {
            "xref": "x1",
            "yref": "y1",
            "text": "<b>True Positive:" + str(TP),
            "y": "Positive",
            "x": "Positive",
            "font": {
                "color": "black"
            },
            "showarrow": False
        },
        {
            "xref": "x1",
            "yref": "y1",
            "text": "<b>True Negative:" + str(TN),
            "y": "Negative",
            "x": "Negative",
            "font": {
                "color": "black"
            },
            "showarrow": False
        },
        {
            "xref": "x1",
            "yref": "y1",
            "text": "<b>False Positive:" + str(FP),
            "y": "Positive",
            "x": "Negative",
            "font": {
                "color": "black"
            },
            "showarrow": False
        }
    ]
    return(confusion_matrix_dict, trace1, annotations)

def create_pred_label_fun(df, prob_col, threshold, pred_col):
    '''
    create prediction label by threshold
    :param df: prediction pandas dataframe
    :param prob_col: probability column name
    :param threshold: threshold
    :param pred_col: prediction column name
    :return: pandas dataframe
    '''
    df_2 = df.copy()
    df_2[pred_col] = df_2[prob_col].apply(lambda x: 1 if x > threshold else 0)
    return df_2


def create_roc_trace(df, label_col, prob_col, legend_name, visible=True):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(df[label_col], df[prob_col])
    roc_auc = auc(false_positive_rate, true_positive_rate).round(2)
    trace = go.Scatter(
        x=false_positive_rate,
        y=true_positive_rate,
        mode='lines',
        line=dict(width=2),
        name=f'{legend_name} (AUC = {roc_auc})',
        visible=visible)
    return trace

def create_roc_trace_withAUC(df, label_col, prob_col, legend_name, visible=True):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(df[label_col], df[prob_col])
    roc_auc = auc(false_positive_rate, true_positive_rate).round(2)
    trace = go.Scatter(
        x=false_positive_rate,
        y=true_positive_rate,
        mode='lines',
        line=dict(width=2),
        name=f'{legend_name} (AUC = {roc_auc})',
        visible=visible)
    return trace, roc_auc


def j(trace_list, height=600, width=840, title='<b>Receiver Operating Characteristic Curve'):
    trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='navy', width=2, dash='dash'),
                        showlegend=False)
    data = trace_list + [trace2]
    layout = go.Layout(title=title,
                       height=height,
                       width=width,
                       xaxis=dict(title='False Positive Rate',
                                  range=[0, 1],
                                  tick0=0,
                                  dtick=0.1),
                       yaxis=dict(title='True Positive Rate',
                                  range=[0, 1],
                                  tick0=0,
                                  dtick=0.1))
    fig = go.Figure(data=data, layout=layout)
    return plotly.offline.iplot(fig, filename='overlaid histogram')


def create_ts_pred_acutal(df_train, df_test, date, label, pred, title="Prediction/Actual", dash_type=False):
    trace1 = go.Scatter(x=df_test[date], y=df_test[pred],
                    mode='lines+markers',
                    name='Prediction_Test', visible=True)

    trace2 = go.Scatter(x=df_test[date], y=df_test[label],
                    mode='lines+markers',
                    name='Actual_Test',visible=True)

    trace3 = go.Scatter(x=df_train[date], y=df_train[pred],
                    mode='lines+markers',
                    name='Prediction_Train', visible=False)

    trace4 = go.Scatter(x=df_train[date], y=df_train[label],
                        mode='lines+markers',
                        name='Actual_Train', visible=False)

    data =[trace1, trace2, trace3, trace4]

    buttons = []
    visible_list = visible_true_false_list(4, 2)
    buttons.append({'label': "Test",'method': 'update', 'args': [{'visible': visible_list[0]}]})
    buttons.append({'label': "Train",'method': 'update', 'args': [{'visible': visible_list[1]}]})
    updatemenus = list([
                    dict(type="buttons",
                         active=-1,
                         x=0.0,
                         xanchor='left',
                         y=1.33,
                         yanchor='top',
                         direction='left',
                         buttons=buttons,
                         )
                ])

    layout = go.Layout(
          updatemenus=updatemenus,
        xaxis=dict(
                title='Date',
            ),
            yaxis=dict(
                title='Value',
                showline=True,
            ),
            title=f"<b><br>{title}<b>"
        )
    if dash_type:
        return data, layout
    else:
        fig = go.Figure(data=data, layout=layout)
        return iplot(fig)


def create_confusion_heatmap_trace_from_dict(input_dict, visible=False):
    FN, TN, TP, FP = input_dict["FN"], input_dict["TN"], input_dict["TP"], input_dict["FP"]
    trace = go.Heatmap(z=[[FN, TN],
                          [TP, FP]],
                       x=['Positive', 'Negative'],
                       y=['Negative', 'Positive'],
                       colorscale=[
                           [1.0, 'rgb(165,0,38)'], [0.88, 'rgb(215,48,39)'], [0.78, 'rgb(244,109,67)'],
                           [0.66, 'rgb(253,174,97)'],[0.55, 'rgb(254,224,144)'], [0.44, 'rgb(224,243,248)'],
                           [0.33, 'rgb(171,217,233)'], [0.22, 'rgb(116,173,209)'], [0.11, 'rgb(69,117,180)'],
                           [0, 'rgb(49,54,149)']], visible=visible)
    return trace


def create_confusion_heatmap_trace(df, label_col, pred_col, visible=False):
    """
    create heatmap trace based on model prediction pandas dataframe
    :param df: prediction dataframe; Pandas Dataframe
    :param label_col: label column name; string
    :param pred_col: prediction column name; String
    :param visible: visible; Boolean
    :return: (plotly trace, annotation dictionary); tuple
    """
    TN, FP, FN, TP = confusion_matrix(df[label_col], df[pred_col]).ravel()
    trace = go.Heatmap(
        z=[[FN, TN], [TP, FP]],
        x=['Positive', 'Negative'],
        y=['Negative', 'Positive'],
        colorscale=[[1.0, 'rgb(165,0,38)'], [0.88, 'rgb(215,48,39)'], [0.78, 'rgb(244,109,67)'],
            [0.66, 'rgb(253,174,97)'], [0.55, 'rgb(254,224,144)'], [0.44, 'rgb(224,243,248)'],
            [0.33, 'rgb(171,217,233)'],[0.22, 'rgb(116,173,209)'], [0.11, 'rgb(69,117,180)'],
            [0, 'rgb(49,54,149)']], visible=visible)
    return trace


def confusion_heatmap_plot(trace_list, tab_list, button_type=None,
                           true_axis_title='', pred_axis_title=''):
    """
    create iplot based on heatmap trace/trace_list
    :param trace_list: plotly heatmap trace/ trace list
    :param annotations_list: heatmap annotation/ heatmap annotation list
    :param tab_list: name of the tab/dropdown; list
    :param button_type: buttons, dropdown,(None if only one trace)
    :param true_axis_title: axis title; string
    :param pred_axis_title: axis title; string
    :return: iplot object
    """

    annotations_list = []
    trace_list[0].visible = True
    for i in range(len(trace_list)):
        temp_trace = trace_list[i]
        FN, TN, TP, FP = temp_trace.z[0][0], temp_trace.z[0][1], temp_trace.z[1][0], temp_trace.z[1][1]
        temp_annotation = [{
            'showarrow': False,
            "text": f"<b>False Negative:{FN}<b>",
            "x": "Positive",
            "xref": "x",
            "y": "Negative",
            "yref": "y"},
            {
                'showarrow': False,
                "text": f"<b>True Negative:{TN}<b>",
                "x": "Negative",
                "xref": "x",
                "y": "Negative",
                "yref": "y"},
            {
                'showarrow': False,
                "text": f"<b>True Positive:{TP}<b>",
                "x": "Positive",
                "xref": "x",
                "y": "Positive",
                "yref": "y"},
            {
                'showarrow': False,
                "text": f"<b>False Positive:{FP}<b>",
                "x": "Negative",
                "xref": "x",
                "y": "Positive",
                "yref": "y"}]
        annotations_list.append(temp_annotation)

    if button_type:
        buttons = []
        var_type = tab_list
        visible_list = visible_true_false_list(len(trace_list), 1)

        for i in range(len(visible_list)):
            temp = {'label': var_type[i], 'method': 'update',
                    'args': [{'visible': visible_list[i]}, {'title.text': f"<b>{tab_list[i]} Confusion Matrix<b>",
                                                            'title.x': 0.5, 'title.xanchor': 'center',
                                                            "annotations": annotations_list[i]}]}
            buttons.append(temp)
        if button_type == "buttons":
            updatemenus = list([
                dict(type=button_type,
                     active=-1,
                     x=0.0,
                     xanchor='left',
                     y=1.33,
                     yanchor='top',
                     direction='right',
                     buttons=buttons,
                     )
            ])
        elif button_type == "dropdown":
            updatemenus = list([
                dict(type=button_type,
                     active=-1,
                     x=0.0,
                     xanchor='left',
                     y=1.33,
                     yanchor='top',
                     direction='down',
                     buttons=buttons,
                     )
            ])

        layout = go.Layout(
            title=dict(
                text=f"<b>{tab_list[0]} Confusion Matrix<b>",
                x=0.5,
                xanchor='center'
            ),
            updatemenus=updatemenus,
            xaxis=dict(title=true_axis_title),
            yaxis=dict(title=pred_axis_title),
            annotations=annotations_list[0]
        )
        data = trace_list
        fig = go.Figure(data=data, layout=layout)
        return fig.show()

    else:
        tab_name = tab_list[0]
        layout = go.Layout(title=dict(
            text=f"<b>{tab_name} Confusion Matrix<b>",
            x=0.5,
            xanchor='center'
        ),
            xaxis=dict(title=true_axis_title),
            yaxis=dict(title=pred_axis_title),
            annotations=annotations_list[0])
        data = trace_list
        fig = go.Figure(data=data, layout=layout)
        return fig.show()


def create_model_evaluation_by_threshold(df, threshold_list, model_name, label_col, prob_col):
    '''
    create threshold df
    :param df: pandas df
    :param threshold_list: threshold list
    :param model_name: model name
    :return: pandas df
    '''
    result = pd.DataFrame()
    for threshold in threshold_list:
        temp_df = create_pred_label_fun(
            df, prob_col, threshold, 'pred_col')
        number_positive_predictions = temp_df.loc[temp_df['pred_col']==1].shape[0]
        model_class = ClassifierModelEvaluation(
            temp_df, model_name, label_col, 'pred_col', prob_col)
        temp_df_2 = model_class.summary
        temp_df_2['threshold'] = threshold
        temp_df_2['n_positive_pred'] = number_positive_predictions
        result = pd.concat([temp_df_2, result], axis=0)
    return result


def create_pr_curve_trace(df, label_col, prob_col, model_name, visible=True):
    '''
    create plotly object for pr_curve
    :param df: prediction pandas dataframe
    :param label_col: label column name
    :param prob_col: probability column name
    :param model_name: model name
    :return: plotly trace object
    '''
    precision, recall, _ = precision_recall_curve(df[label_col], df[prob_col])
    pr_auc = auc(recall, precision)
    trace = go.Scatter(
        x=recall, 
        y=precision,                    
        mode='lines',
        line=dict(width=2),
        name='{} (area = {:.2f})'.format(model_name, pr_auc),
        visible=visible)
    return trace


def create_overlay_pr_curve(trace_list, height=600, width=840, title='<b>PR Curve'):
    '''
    Create Plotly overlay pr cruve graph
    :param trace_list: pr curve trace list
    :return: plotly object
    '''
    data = trace_list
    layout = go.Layout(title=title,
                       height=height,
                       width=width,
                       legend=dict(x=0.9, y=1, font=dict(size=13)),
                       xaxis=dict(title='Recall',
                                  range=[0, 1],
                                  tick0=0,
                                  dtick=0.1),
                       yaxis=dict(title='Precision',
                                  range=[0, 1],
                                  tick0=0,
                                  dtick=0.1))
    fig = go.Figure(data=data, layout=layout)
    return plotly.offline.iplot(fig, filename='overlaid histogram')

def create_overlay_roc_curve(trace_list, height=600, width=840, title='<b>Receiver Operating Characteristic Curve'):
    trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='navy', width=2, dash='dash'),
                        showlegend=False)
    data = trace_list + [trace2]
    layout = go.Layout(title=title,
                       height=height,
                       width=width,
                       xaxis=dict(title='False Positive Rate',
                                  range=[0, 1],
                                  tick0=0,
                                  dtick=0.1),
                       yaxis=dict(title='True Positive Rate',
                                  range=[0, 1],
                                  tick0=0,
                                  dtick=0.1))
    fig = go.Figure(data=data, layout=layout)
    return plotly.offline.iplot(fig, filename='overlaid histogram')

def create_ts_pred_acutal(df_train, df_test, date, label, pred, title="Prediction/Actual", dash_type=False):
    trace1 = go.Scatter(x=df_test[date], y=df_test[pred],
                    mode='lines+markers',
                    name='Prediction_Test', visible=True)

    trace2 = go.Scatter(x=df_test[date], y=df_test[label],
                    mode='lines+markers',
                    name='Actual_Test',visible=True)

    trace3 = go.Scatter(x=df_train[date], y=df_train[pred],
                    mode='lines+markers',
                    name='Prediction_Train', visible=False)

    trace4 = go.Scatter(x=df_train[date], y=df_train[label],
                        mode='lines+markers',
                        name='Actual_Train', visible=False)

    data =[trace1, trace2, trace3, trace4]

    buttons = []
    visible_list = visible_true_false_list(4, 2)
    buttons.append({'label': "Test",'method': 'update', 'args': [{'visible': visible_list[0]}]})
    buttons.append({'label': "Train",'method': 'update', 'args': [{'visible': visible_list[1]}]})
    updatemenus = list([
                    dict(type="buttons",
                         active=-1,
                         x=0.0,
                         xanchor='left',
                         y=1.33,
                         yanchor='top',
                         direction='left',
                         buttons=buttons,
                         )
                ])

    layout = go.Layout(
          updatemenus=updatemenus,
        xaxis=dict(
                title='Date',
            ),
            yaxis=dict(
                title='Value',
                showline=True,
            ),
            title=f"<b><br>{title}<b>"
        )
    if dash_type:
        return data, layout
    else:
        fig = go.Figure(data=data, layout=layout)
        return plotly.offline.iplot(fig)


    
    
def bootstrap_results(df, f, metric_name, label_col='label', prediction_col='prediction', nsamples=1000):
    stats = []
    for b in range(nsamples):
        indices = np.random.randint(df.shape[0], size=int(df.shape[0]/2))
        new_stats = f(df.iloc[indices, :], label_col, prediction_col)[metric_name]
        stats.append(new_stats)
    return np.percentile(stats, (2.5, 50, 97.5))

def bootstrap_auroc(df, label_col, pred_col, nsamples=1000):
    stats = []
    for b in range(nsamples):
        indices = np.random.randint(df.shape[0], size=int(df.shape[0]/2))
        df2 = df.iloc[indices, :]
        false_positive_rate, true_positive_rate, thresholds = roc_curve(df2[label_col], df2[pred_col])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        stats.append(roc_auc)

    return np.percentile(stats, (2.5, 50, 97.5))

def bootstrap_aupr(df, label_col, pred_col, nsamples=1000):
    stats = []
    for b in range(nsamples):
        indices = np.random.randint(df.shape[0], size=int(df.shape[0]/2))
        df2 = df.iloc[indices, :]
        precision, recall, _ = precision_recall_curve(df2[label_col], df2[pred_col])
        pr_auc = auc(recall, precision)
        stats.append(pr_auc)

    return np.percentile(stats, (2.5, 50, 97.5))


def get_performance_metrics_from_confusion_matrix_dict(confusion_matrix_dict):
    TP, FP, TN, FN = confusion_matrix_dict['TP'], confusion_matrix_dict['FP'], confusion_matrix_dict['TN'], confusion_matrix_dict['FN']
    if TP + FN == 0:
        sensitivity = 'N/A'
    else:
        sensitivity = round(float(TP) / (TP + FN), 3)
        
    if TN + FP == 0:
        specificity = 'N/A'
    else:
        specificity = round(float(TN) / (TN + FP), 3)
        
    if TP + FP == 0:
        precision = 'N/A'
    else:
        precision = round(float(TP) / (TP + FP), 3)
        
        
    if precision=='N/A' or sensitivity=='N/A':
        f1_score = 'N/A'
    else:
        f1_score = round((2 * (precision * sensitivity) / (precision + sensitivity)), 3)
        
    accuracy = float(TN + TP) / (TN + FP + FN + TP)
    
    if FP+TN==0:
        fpr='N/A'
    else:
        fpr = float(FP)/(FP+TN)
    return(sensitivity, specificity, precision, f1_score, precision, accuracy, fpr)



class ClassifierModelEvaluation(object):

    def __init__(self, df, model_name=None, label="label", pred_col="pred", prob_col="prob"):
        '''
        :param df: prediction dataframe
        :param label: True label column name
        :param pred_col: prediction column name
        :param prob_col: probability column name
        '''

        self.df = df
        self.sample_size = self.df.shape[0]
        self.label = label
        self.pred_col = pred_col
        self.prob_col = prob_col
        self.model_name = model_name
        self.TN, self.FP, self.FN, self.TP = confusion_matrix(df[self.label], df[self.pred_col]).ravel()
        self.sensitivity = self.model_eval_sensitivity()
        self.specificity = self.model_eval_specificity()
        self.precision = self.model_eval_precision()
        self.f1_score = self.model_eval_f1_score()
        self.accuracy = self.model_eval_accuracy()
        self.auc = self.auc_fun(self.df, self.label, self.prob_col)
        # Precision or positive predictive value
        self.PPV = round(float(self.TP) / (self.TP + self.FP), 3)
        # Negative predictive value
        self.NPV = round(float(self.TN) / (self.TN + self.FN), 3)
        # Fall out or false positive rate
        self.FPR = round(float(self.FP) / (self.FP + self.TN), 3)
        # False negative rate
        self.FNR = round(float(self.FN) / (self.TP + self.FN), 3)
        # False discovery rate
        # check here: https://en.wikipedia.org/wiki/False_discovery_rate
        self.FDR = round(float(self.FP) / (self.TP + self.FP), 3)
        self.confusion_df = self.create_confusion_df()
        if model_name:
            self.summary = self.model_summary(model_name)
        else:
            self.summary = self.model_summary()

    def create_confusion_df(self):
        try:
            confusion_df = pd.DataFrame({"TN": [self.TN], "FP": [self.FP], "FN": [self.FN], "TP": [self.TP], "sample_size": [self.sample_size]})
        except:
            confusion_df = pd.DataFrame({"TN": [], "FP": [], "FN": [], "TP": [], "sample_size": []})
        return confusion_df

    def model_eval_sensitivity(self):
        '''
        :param df: prediction dataframe
        :param label_col: True label column name
        :param pred_col: prediction column name
        :return: float
        '''
        sensitivity = round(float(self.TP) / (self.TP + self.FN), 3)
        return sensitivity

    def model_eval_specificity(self):
        '''
        :param df: prediction dataframe
        :param label_col: True label column name
        :param pred_col: prediction column name
        :return: float
        '''
        specificity = round(float(self.TN) / (self.TN + self.FP), 3)
        return specificity

    def model_eval_precision(self):
        '''
        :param df: prediction dataframe
        :param label_col: True label column name
        :param pred_col: prediction column name
        :return: float
        '''
        precision = round((float(self.TP) / (self.TP + self.FP)), 3)
        return precision

    def model_eval_f1_score(self):
        '''
        :param df: prediction dataframe
        :param label_col: True label column name
        :param pred_col: prediction column name
        :return: float
        '''

        sensitivity = self.sensitivity
        precision = float(self.TP) / (self.TP + self.FP)
        f1_score = round((2 * (precision * sensitivity) / (precision + sensitivity)), 3)
        return f1_score

    def model_eval_accuracy(self):
        '''
        :param df: prediction dataframe
        :param label_col: True label column name
        :param pred_col: prediction column name
        :return: float
        '''
        accuracy = float(self.TN + self.TP) / (self.TN + self.FP + self.FN + self.TP)
        return accuracy

    def auc_fun(self, df, label, prob_col="prob"):
        '''
        :param df: prediction dataframe
        :param label: True label column name
        :param prob_col: probability column name
        :return: float
        '''
        false_positive_rate, true_positive_rate, thresholds = roc_curve(df[label], df[prob_col])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        return roc_auc

    def generate_model_performance_table(
            self,
            threshold=0.5,
            add_threshold=True,
            add_total_sample=True,
            add_true_negative=True,
            add_false_positive=True,
            add_False_negative=True,
            add_true_positive=True,
            add_sensitivity=True,
            add_specificity=True,
            add_precision=True,
            add_accuracy=True,
            add_f1_score=True,
            add_PPV=False,
            add_NPV=False,
            add_FPR=False,
            add_FNR=False,
            add_FDR=False
    ):
        '''
        :param df: final dataframe which has the true label and predictions column
        :param threshold: Set Threshold
        :param label_col: True label column in the dataframe
        :param pred_col: Prediction column in the dataframe
        the default list of metrics is: 'Threshold', 'Total_Sample', 'True_Negative', 'False_Positive',
                  'False_Negative', 'True_Positive', 'Sensitivity', 'Specificity', "Precision","Accuracy",  "F1_score"
        :param add_PPV: flag, if True, we want to add PPV as a metric in the performance table
        :param add_NPV: flag, if True, we want to add NPV as a metric in the performance table
        :param add_FPR: flag, if True, we want to add FPR as a metric in the performance table
        :param add_FNR: flag, if True, we want to add FNR as a metric in the performance table
        :param add_FDR: flag, if True, we want to add FDR as a metric in the performance table
        :return: Performance Matrix
        '''
        templist = []
        total_patients = self.TN + self.FP + self.FN + self.TP

        metrics_dict = {
            'Threshold': [add_threshold, threshold],
            'Total_Sample': [add_total_sample, total_patients],
            'True_Negative': [add_true_negative, self.TN],
            'True_Positive': [add_true_positive, self.TP],
            'False_Positive': [add_false_positive, self.FP],
            'False_Negative': [add_False_negative, self.FN],
            'Sensitivity': [add_sensitivity, self.sensitivity],
            'Specificity': [add_specificity, self.specificity],
            'Precision': [add_precision, self.precision],
            'Accuracy': [add_accuracy, self.accuracy],
            'F1_score': [add_f1_score, self.f1_score],
            'Positive_Predicted_Value': [add_PPV, self.PPV],
            'Negative_Predicted_Value': [add_NPV, self.NPV],
            'False_Positive_Rate': [add_FPR, self.FPR],
            'False_Negative_Rate': [add_FNR, self.FNR],
            'False_Discovery_Rate': [add_FDR, self.FDR]
        }
        labels = [k for k, v in metrics_dict.items() if v[0]]
        metrics = [v[1] for k, v in metrics_dict.items() if v[0]]
        templist.append(metrics)
        perf_matrix = pd.DataFrame.from_records(templist, columns=labels)
        return perf_matrix


    def model_summary(self, model_name=None):
        """
        create model evaluation summary dataframe
        :param model_name: model name
        :return: pandas df
        """
        if model_name:
            summary_df = pd.DataFrame({'sample size': [self.sample_size], "sensitivity": [self.sensitivity],
                                       "specificity": [self.specificity],
                                       "f1_score": [self.f1_score], "accuracy": [self.accuracy],
                                       "precision": [self.precision],
                                       "auc": [self.auc], "model_name": [model_name]})
        else:
            summary_df = pd.DataFrame({"sensitivity": [self.sensitivity], "specificity": [self.specificity],
                                       "f1_score": [self.f1_score], "accuracy": [self.accuracy],
                                       "precision": [self.precision],
                                       "auc": [self.auc]})

        return summary_df

    def create_confusion_heatmap(self, model_name=None, true_axis_title='', pred_axis_title=''):
        '''
        create confusion matrix heatmap
        :param model_name: model name
        :param true_axis_title: x-axis title
        :param pred_axis_title: y-axis title
        :return: plotly object
        '''
        trace = create_confusion_heatmap_trace(self.df, self.label, self.pred_col)
        if model_name:
            graph = confusion_heatmap_plot([trace], [model_name], button_type=None,
                                           true_axis_title=true_axis_title, pred_axis_title=pred_axis_title)
            return graph
        else:
            graph = confusion_heatmap_plot([trace], [self.model_name], button_type=None,
                                           true_axis_title=true_axis_title, pred_axis_title=pred_axis_title)
            return graph


class RegressionModelEvaluation(object):
    def __init__(self, df, pred_col, label_col, model_name):
        '''
        :param df: pandas regression prediction DF, pandas DF
        :param pred_col: prediction column name, String
        :param label_col: label column name, String
        :param model_name: model name, String
        '''
        self.df = df.copy()
        self.sample_size = df.shape[0]
        self.pred_col = pred_col
        self.label_col = label_col
        self.model_name = model_name
        self.error_rate()
        self.rmse = self.mean_squared_error()
        self.mae = self.mean_absolute_error()
        self.r2_score = self.r2_score()
        self.count_by_int_category = df.groupby([pred_col, label_col]).size().to_frame(name="datapoint_count").reset_index()

        self.count_datapoints = pd.merge(
            df,
            self.count_by_int_category,
            on=[pred_col, label_col],
            how='left'
        )
        # normalize size for markers
        min_size = self.count_datapoints['datapoint_count'].min()
        max_size = self.count_datapoints['datapoint_count'].max()
        if max_size==min_size:
            self.count_datapoints['norm_size'] = 10
        else:
            self.count_datapoints['norm_size'] = (self.count_datapoints['datapoint_count'] - min_size) / (max_size - min_size)
            self.count_datapoints['norm_size'] = self.count_datapoints['norm_size'] * 50 + 10



        # normalize size for markers
        min_size = self.count_by_int_category['datapoint_count'].min()
        max_size = self.count_by_int_category['datapoint_count'].max()
        if max_size == min_size:
            self.count_by_int_category['norm_size'] = 10
        else:
            self.count_by_int_category['norm_size'] = (self.count_by_int_category['datapoint_count'] - min_size) / (
                        max_size - min_size)
            self.count_by_int_category['norm_size'] = self.count_by_int_category['norm_size'] * 50 + 10


    def error_rate(self):
        self.df['error'] = (self.df[self.pred_col] - self.df[self.label_col]) / self.df[self.label_col]
        self.df['abs_error'] = abs(self.df[self.pred_col] - self.df[self.label_col]) / self.df[self.label_col]
        self.df['accuracy'] = 1 - self.df['abs_error']

    def mean_squared_error(self):
        rmse = sqrt(mean_squared_error(
            self.df[self.label_col], self.df[self.pred_col]))
        return rmse

    def mean_absolute_error(self):
        return mean_absolute_error(self.df[self.label_col], self.df[self.pred_col])

    def r2_score(self):
        return r2_score(self.df[self.label_col], self.df[self.pred_col])

    def r2_score_adjusted(self, n_feature):
        '''
        :param n_feature: number of features in model: int
        :return: r2 adjusted score: float
        '''
        r_adjusted = 1 - (1 - self.r2_score) * (
                (self.sample_size - 1) / (self.sample_size - n_feature - 1))
        return r_adjusted

    def model_summary(self, n_feature=None):
        '''
        :param n_feature: number of features in model: int
        :return: model summary, pandas df
        '''
        if n_feature:
            summary_df = pd.DataFrame(
                {'SampleSize': [self.sample_size], "ModelName": [self.model_name],
                 "RootMeanSquareError": [self.rmse], "MeanAbsoluteError": [self.mae],
                 "R_Square": [self.r2_score], "ErrorRate": [self.df['abs_error'].mean()],
                 "Accuracy": [self.df['accuracy'].mean()],
                 "R_Square_Adjust": [self.r2_score_adjusted(n_feature)]}).round(3)
        else:
            summary_df = pd.DataFrame(
                {'SampleSize': [self.sample_size], "ModelName": [self.model_name],
                 "RootMeanSquareError": [self.rmse], "MeanAbsoluteError": [self.mae],
                 "ErrorRate": [self.df['abs_error'].mean()], "Accuracy": [self.df['accuracy'].mean()],
                 "R_Square": [self.r2_score]}).round(3)
        return summary_df

    def create_r2_graph(self, x, y, dash_type=False, visible=True):
        '''
        :param x: label column name String
        :param y: prediction column name String
        :param dash_type: return dash type format or not: boolean
        :return: plotly object
        '''
        def create_regression_line(x, y):
            x_1, y_1 = x.to_numpy(), y.to_numpy()
            x_min = min(min(x_1), min(y_1))
            x_max = max(max(x_1), max(y_1))
            x_interval = np.linspace(x_min, x_max, 100).tolist()
            y_interval = x_interval
            return x_interval, y_interval

        x_line, y_line = create_regression_line(self.df[x], self.df[y])

        trace1 = go.Scatter(x=self.df[x], y=self.df[y],
                            mode='markers',
                            name='Actual/Predicted', visible=visible)
        trace2 = go.Scatter(x=x_line, y=y_line,
                            mode='lines',
                            name='Regression Line', visible=visible)

        r2 = round(self.r2_score, 2)
        data = [trace1, trace2]
        layout = go.Layout(
            annotations=[
                dict(
                    x=max(x_line),
                    y=max(y_line),
                    xref="x",
                    yref="y",
                    text=f"R-Squared Score = {r2}",
                    arrowhead=8,
                    ax=0,
                    ay=-40
                )
            ],
            xaxis=dict(
                title='Actual number of completed chemo infusions',

            ),
            yaxis=dict(
                title='Predicted number of completed chemo infusions',

            ),
            title="<b><br>R-Squared Graph<b>"
        )
        if dash_type:
            return data, layout
        else:
            fig = go.Figure(data=data, layout=layout)
            return iplot(fig)

    def create_r2_graph_integer(
        self, 
        x, 
        y, 
        dash_type=False, 
        visible=True, 
        xaxis_title = '', 
        yaxis_title = ''
    ):
        '''
        :param x: label column name String
        :param y: prediction column name String
        :param dash_type: return dash type format or not: boolean
        :param xaxis_title: title for the x axis
        :param yaxis_title: title for the y axis
        :param hover_name_x: name of the entity we want to show in hover text for x labels
        :param hover_name_y: name of the entity we want to show in hover text for y labels
        :return: plotly object
        '''

        def create_regression_line(x, y):
            x_1, y_1 = x.to_numpy(), y.to_numpy()
            x_min = min(min(x_1), min(y_1))
            x_max = max(max(x_1), max(y_1))
            x_interval = np.arange(x_min, x_max+1).tolist()
            y_interval = x_interval
            return x_interval, y_interval

        x_line, y_line = create_regression_line(self.count_by_int_category[x], self.count_by_int_category[y])

        trace1 = go.Scatter(x=self.count_by_int_category[x], y=self.count_by_int_category[y],
                            mode='markers',
                            marker=dict(
                                size=self.count_by_int_category['norm_size']),
                            name='Actual/Predicted',
                            hovertext=[
                                f"Count for ({self.count_by_int_category[x][i]}, {self.count_by_int_category[y][i]}): {self.count_by_int_category['datapoint_count'][i]}" for i in range(len(self.count_by_int_category))
                                ],
                            hoverinfo='text',
                            visible=visible
                           )
        trace2 = go.Scatter(x=x_line, y=y_line,
                            mode='lines',
                            name='Regression Line', visible=visible
                           )

        r2 = round(self.r2_score, 2)
        data = [trace1, trace2]
        layout = dict(
            height=600, width=800,
            title= 'R-Squared Graph<br><sup>(the diameters of data points are proportionate to the number of matching predictions)</sup>',
            legend=dict(x=1, y=1, font=dict(size=12)), 
            xaxis=dict(title=xaxis_title, dtick="L1.0"),
            yaxis=dict(title=yaxis_title, dtick="L1.0"),
            hovermode='closest')
        if dash_type:
            return data, layout
        else:
            fig = go.Figure(data=data, layout=layout)
            return iplot(fig)

    def create_r2_adjusted_graph(self, n_feature, x, y, dash_type = False, visible = True):
        '''
        :param n_feature: number of features in model: int
        :param x: label column name String
        :param y: prediction column name String
        :param dash_type: return dash type format or not: boolean
        :return: plotly object
        '''
        def create_regression_line(x, y):
            x_1, y_1 = x.to_numpy(), y.to_numpy()
            x_min = min(min(x_1), min(y_1))
            x_max = max(max(x_1), max(y_1))
            x_interval = np.linspace(x_min, x_max, 100).tolist()
            y_interval = x_interval
            return x_interval, y_interval

        x_line, y_line = create_regression_line(self.df[x], self.df[y])

        trace1 = go.Scatter(x=self.df[x], y=self.df[y],
                            mode='markers',
                            name='Actual/Predicted', visible=visible)
        trace2 = go.Scatter(x=x_line, y=y_line,
                            mode='lines',
                            name='Regression Line', visible=visible)

        r2_adjusted = round(self.r2_score_adjusted(n_feature), 2)
        data = [trace1, trace2]
        layout = go.Layout(
            annotations=[
                dict(
                    x=max(x_line),
                    y=max(y_line),
                    xref="x",
                    yref="y",
                    text=f"R^2 Adjusted Score = {r2_adjusted}",
                    arrowhead=8,
                    ax=0,
                    ay=-40
                )
            ],
            xaxis=dict(
                title='Actual',

            ),
            yaxis=dict(
                title='Predicted',

            ),
            title="<b><br>R^2 Adjusted Graph<b>"
        )
        if dash_type:
            return data, layout
        else:
            fig = go.Figure(data=data, layout=layout)
            return iplot(fig)
