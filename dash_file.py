"""
Mirza N Ahmed
DS3500 / WintourWardrobe
Date Created: 4/7/2023 / Date Last Updated: 4/10/2023

File name: dash_file
Desc: Uses the trained Machine Learning model to generate a Plotly Dashboard
"""

# Imports
import base64
import io
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import ml_model as fr
from PIL import Image

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])  # applies CSS stylesheet

# Create the layout
app.layout = dbc.Container([

    # 1) Add a row on top with the title
    dbc.Row([
        dbc.Col(html.H1("Wintour Wardrobe", className="text-center"),

                # styling: added a border
                width=12, style={'border': '1px solid #c4c7c9',
                                 'padding': '10px',
                                 'font-family': 'Noto Sans'})
    ]),

    # Create a row with two columns
    dbc.Row([

        # Column 1 - Upload an image
        dbc.Col([

            # instructions
            html.Div(
                "Upload the photo of a clothing item of your choice."
                "I'll suggest similar options for you to consider.",
                className="text-center mb-3"),

            # upload option
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select an Image')
                ]),
                className="d-flex justify-content-center",
                style={
                    'width': '95%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                }
            )
            # added some styling (border, padding)
        ], width=4, style={'border': '1px solid #c4c7c9', 'padding': '10px'}),

        # Column 2 - Display the uploaded image
        dbc.Col([
            html.Div(id='uploaded-image-label', children="Uploaded image:", className="d-flex justify-content-center"),
            html.Div(style={"height": "5px"}),  # add some height for the gap
            html.Div(id='uploaded-image', className="d-flex justify-content-center")
            # add styling (border, padding, display)
        ], width=4,
            style={'border': '1px solid #c4c7c9', 'padding': '10px'}),

        # Column 3 - Display the prediction output
        dbc.Col([
            html.Div(className="d-flex flex-column justify-content-center align-items-center", children=[
                html.Div(id='output-prediction', style={'width': '100%'}),
                html.Div(style={"height": "5px"}),  # add some height for the gap
                # more button to control offset
                html.Button('More', id='more-button', n_clicks=0, className="btn-primary")
            ]),
        ], width=4, style={'border': '1px solid #c4c7c9', 'padding': '10px'})

        # styling: some padding before the bottom row
    ], style={'padding': '50px 100px'}),

    # addition row at the bottom (description, limitation - for the poster session)
    dbc.Row([
        dbc.Col([
            html.H3("Short Description", className="text-center"),
            html.Ul([
                html.Li("Lorem ipsum"),
                html.Li("Lorem ipsum"),
                html.Li("Lorem ipsum")
            ], className="list-unstyled text-center")
        ], width=6),

        dbc.Col([
            html.H3("Limitations", className="text-center"),
            html.Ul([
                html.Li("Lorem ipsum"),
                html.Li("Lorem ipsum"),
                html.Li("Lorem ipsum")
            ], className="list-unstyled text-center")
        ], width=6)

        # overall border for the row
    ], style={'border': '1px solid #c4c7c9', 'padding': '10px'})

    # styling for the overall bg color and text
], style={'backgroundColor': '#36454F', 'color': '#ebeced'})


# Callback to handle image upload and display the prediction
@app.callback(
    [Output('output-prediction', 'children'),
     Output('uploaded-image', 'children')],

    # the order matters
    Input('upload-image', 'contents'),
    Input('more-button', 'n_clicks'),
    State('upload-image', 'filename')
)
def update_output(contents, n_clicks, filename):
    """
    Call back function that handles image uploads and generates prediction using the ML model
    :param contents: (str) base64 encoded contents of uploaded image
    :param filename: (str) filename of uploaded image
    :param n_clicks: (Int) number of times the more button has been clicked
    :return: (list) html components needed to display return
    """

    uploaded_image = None
    output_prediction = []

    if n_clicks is None:
        n_clicks = 0

    if contents is not None:

        # splitting the base64 encoded content string to get the image bytes
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        uploaded_image = html.Img(src=contents, style={'width': '100%', 'max-width': '200px'})

        # writing the image to a file
        with open(filename, 'wb') as f:
            f.write(decoded)

        # using the training model to make the predictions
        predicted_class, predicted_title = fr.recommender.predict(filename)

        # # obtain the sample images to display
        # sample_images = fr.get_sample_images(predicted_class)

        # Get the next batch of sample images using the n_clicks as an offset
        sample_images, num_returned_images = fr.get_sample_images(predicted_class, num_samples=9, offset=9 * n_clicks)

        # in case we run out of sample images (button clicked too many times)
        if num_returned_images == 0:
            return "No more sample images available", uploaded_image

        # Encode sample images as base64 and create (HTML) image tags
        encoded_images = []
        for img in sample_images:
            img = Image.fromarray(img)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            encoded_images.append(
                html.Img(src=f"data:image/png;base64,{img_str}", style={'width': '28%', 'margin': '1%'}))

        # return the predicted class, title, encoded sample images as HTML Divs
        # return [
        output_prediction = [
            html.Div(className="d-flex justify-content-center flex-column align-items-center", children=[html.Div(
                [html.H5(f"Predicted class: {predicted_class}", style={'font-family': 'Calibri'}),
                 html.H3(f"Predicted title: {predicted_title}", style={'font-family': 'Calibri'})],
                className="text-center mb-3"),
                html.Div(encoded_images, className="d-flex justify-content-center flex-wrap",
                         style={'width': '90%'}), ])
        ]
        return output_prediction, uploaded_image

    # if no image has been uploaded yet, return an empty list
    # return []
    return output_prediction, uploaded_image


if __name__ == '__main__':
    app.run_server(debug=True)
