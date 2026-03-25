import dash
from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION
base_folder = "/home/dlipsey/MITLincolnLabs/MIT_LL_data"
reg_known_folder = f"{base_folder}/cond_enc_1234e1e2"
reg_novel_folder = f"{base_folder}/cond_enc_1234e1e2_super_test"

bin_val = 1
threshold_val = 0.05
loop_num = 0
bin_part = str(bin_val).replace('.', '_')
threshold_part = str(threshold_val).replace('.', '_')
known_file = f"{reg_known_folder}/cond_enc_bin{bin_part}_thresh{threshold_part}_df_spectra_loop{loop_num}.parquet"
novel_file = f"{reg_novel_folder}/super_test_cond_enc_bin{bin_part}_thresh{threshold_part}_df_spectra_loop{loop_num}.parquet"
smiles_col = "SMILES_spectra"

def pad_or_truncate(embeddings, target_dim):
    if embeddings.shape[1] < target_dim:
        return np.hstack([embeddings, np.zeros((embeddings.shape[0], target_dim - embeddings.shape[1]))])
    elif embeddings.shape[1] > target_dim:
        return embeddings[:, :target_dim]
    return embeddings

def sample_spectra_per_chemical(df, smiles_col, max_per_smiles, seed):
    selected_rows = []
    for smile in df[smiles_col].unique():
        smile_rows = df[df[smiles_col] == smile]
        if len(smile_rows) > max_per_smiles:
            selected_rows.extend(smile_rows.sample(max_per_smiles, random_state=seed).index)
        else:
            selected_rows.extend(smile_rows.index)
    return df.loc[selected_rows]

def load_data(file_path, smiles_col, max_smiles, max_spectra, filter_train=None, required_smiles=None, seed=None):
    try:
        df = pd.read_parquet(file_path)
        if filter_train is not None and 'train' in df.columns:
            df = df[df['train'] == filter_train]
        embedding_cols = [col for col in df.columns if col.startswith('cond_emb_')
                          and 'morgan' not in col.lower() and 'filtered' not in col.lower()]
        if not embedding_cols or len(df) == 0:
            return None, None, None, None, set()
        if required_smiles is not None and len(required_smiles) > 0:
            df = df[df[smiles_col].isin(required_smiles)]
        else:
            unique_smiles = pd.unique(df[smiles_col])
            if len(unique_smiles) > max_smiles:
                if seed is not None:
                    np.random.seed(seed)
                selected_smiles_subset = np.random.choice(unique_smiles, max_smiles, replace=False)
                df = df[df[smiles_col].isin(selected_smiles_subset)]
        df = sample_spectra_per_chemical(df, smiles_col, max_spectra, seed)
        embeddings = df[embedding_cols].values
        smiles = df[smiles_col].values
        # Use log LD50 values directly and exponentiate for display
        true_vals = df['Response'].values if 'Response' in df.columns else [None] * len(df)
        pred_vals = df['cond_tox_pred'].values if 'cond_tox_pred' in df.columns else [None] * len(df)
        return embeddings, smiles, true_vals, pred_vals, set(smiles)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, set()

def make_pca_plot(
    max_smiles=20,
    max_spectra=10,
    color_by_smiles=True,
    novel_chemicals_black=False,
    include_novel_chemicals=True,
    include_known_chemicals=True,
    include_training=True,
    random_seed=42
):
    # Load sets
    known_emb, known_smiles, known_true, known_pred, known_smiles_set = None, None, None, None, set()
    if include_known_chemicals:
        known_emb, known_smiles, known_true, known_pred, known_smiles_set = \
            load_data(known_file, smiles_col, max_smiles, max_spectra, filter_train=0, seed=random_seed)
    train_emb, train_smiles, train_true, train_pred, _ = None, None, None, None, set()
    if include_training:
        train_emb, train_smiles, train_true, train_pred, _ = \
            load_data(known_file, smiles_col, max_smiles, max_spectra, filter_train=1, required_smiles=known_smiles_set if include_known_chemicals else None, seed=random_seed)
    novel_emb, novel_smiles, novel_true, novel_pred, _ = None, None, None, None, set()
    if include_novel_chemicals:
        novel_emb, novel_smiles, novel_true, novel_pred, _ = \
            load_data(novel_file, smiles_col, max_smiles, max_spectra, seed=random_seed)

    embedding_dim = None
    for arr in [novel_emb, known_emb, train_emb]:
        if arr is not None:
            embedding_dim = arr.shape[1]
            break
    for arr_lst in [[known_emb], [train_emb], [novel_emb]]:
        for idx, arr in enumerate(arr_lst):
            if arr is not None and embedding_dim is not None:
                arr_lst[idx] = pad_or_truncate(arr, embedding_dim)
    # Data stitching
    trace_types = []
    if include_training and train_emb is not None:
        trace_types.append(('Training', train_emb, train_smiles, train_true, train_pred))
    if include_known_chemicals and known_emb is not None:
        trace_types.append(('Known Chemicals', known_emb, known_smiles, known_true, known_pred))
    if include_novel_chemicals and novel_emb is not None:
        trace_types.append(('Novel Chemicals', novel_emb, novel_smiles, novel_true, novel_pred))
    all_embeddings, all_smiles, all_true_vals, all_pred_vals, all_dataset_types = [], [], [], [], []
    for dtype, emb, smiles, tru, pre in trace_types:
        all_embeddings.extend(emb)
        all_smiles.extend(smiles)
        all_true_vals.extend(tru)
        all_pred_vals.extend(pre)
        all_dataset_types.extend([dtype]*len(smiles))
    if not all_embeddings:
        return go.Figure()
    embeddings_matrix = np.array(all_embeddings)
    smiles_array = np.array(all_smiles)
    true_vals_array = np.array(all_true_vals)
    pred_vals_array = np.array(all_pred_vals)
    dataset_types_array = np.array(all_dataset_types)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_matrix)
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings_scaled)
    unique_smiles = list(set(smiles_array))
    turbo_colors = pc.sample_colorscale("turbo", np.linspace(0, 1, len(unique_smiles)))
    smiles_to_color = {smile: color for smile, color in zip(unique_smiles, turbo_colors)}
    current_novel_smiles_set = set(novel_smiles) if novel_smiles is not None else set()
    traces = []
    for dtype in ['Training','Known Chemicals','Novel Chemicals']:
        mask = dataset_types_array==dtype
        if not mask.any(): continue
        if novel_chemicals_black:
            color_arr = []
            for idx, flag in enumerate(mask):
                if not flag: continue
                smile = smiles_array[idx]
                if dtype == "Novel Chemicals":
                    color_arr.append("black")
                else:
                    color_arr.append(smiles_to_color[smile] if color_by_smiles else 'black')
        else:
            color_arr = [smiles_to_color[smiles_array[idx]] if color_by_smiles else 'black' 
                         for idx, flag in enumerate(mask) if flag]
        subset_hover = []
        for idx, flag in enumerate(mask):
            if not flag: continue
            smile = smiles_array[idx]
            v_true = true_vals_array[idx]
            v_pred = pred_vals_array[idx]
            tp = dtype
            # Use np.exp for LD50
            true_ld50 = np.exp(v_true) if v_true is not None else 'N/A'
            pred_ld50 = np.exp(v_pred) if v_pred is not None else 'N/A'
            txt = (
                f"SMILES: {smile}"
                f"<br>True LD50: {true_ld50:.3f}" if v_true is not None else "<br>True LD50: N/A"
            )
            txt += (
                f"<br>Predicted LD50: {pred_ld50:.3f}" if v_pred is not None else "<br>Predicted LD50: N/A"
            )
            subset_hover.append(txt)
        marker_map = {
            'Training': 'triangle-up',
            'Known Chemicals': 'square',
            'Novel Chemicals': 'circle',
        }
        marker = marker_map[dtype]
        alpha = 1.0 if dtype=="Novel Chemicals" else 0.7
        traces.append(
            go.Scatter(
                x=embeddings_2d[mask,0],
                y=embeddings_2d[mask,1],
                mode='markers',
                name=dtype,
                marker=dict(
                    symbol=marker,
                    size=8,
                    color=color_arr,
                    opacity=alpha,
                    line=dict(width=0.5,color='DarkSlateGrey')
                ),
                text=subset_hover,
                hovertemplate='%{text}<extra></extra>',
                legendgroup=dtype,
                showlegend=True,
            )
        )
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(
            text=f'2D PCA of Regression Encoder Embeddings (Bin={bin_val}, Threshold={threshold_val}, Loop={loop_num})',
            font=dict(size=18, color='black'), x=0.5, xanchor='center'
        ),
        xaxis=dict(
            title='Principal Component 1 (PC1)', title_font=dict(size=14),
            showgrid=True, gridwidth=0.5, gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False, showline=True, linewidth=3,
            linecolor='black', mirror=True
        ),
        yaxis=dict(
            title='Principal Component 2 (PC2)', title_font=dict(size=14),
            showgrid=True, gridwidth=0.5, gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False, showline=True, linewidth=3,
            linecolor='black', mirror=True
        ),
        hovermode='closest',
        height=950, width=1200,
        margin=dict(l=80, r=150, t=80, b=120),
        legend=dict(
            title=dict(text="Dataset Types", font=dict(size=14, color='black')),
            x=0, y=1,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black', borderwidth=2, font=dict(size=12)
        ),
        plot_bgcolor='rgba(240, 240, 250, 0.5)'
    )
    return fig

# --- Dash UI
max_smiles_values = [5, 10, 15, 20, 30, 40, 50]
max_spectra_values = [1, 3, 5, 7, 10, 15, 20]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("2D PCA of Regression Encoder Embeddings"),
    html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Label("Max SMILES"),
                dcc.Dropdown(
                    id="smiles-dropdown",
                    options=[{"label": str(v), "value": v} for v in max_smiles_values],
                    value=max_smiles_values[2],
                    clearable=False
                ),
            ], md=3),
            dbc.Col([
                dbc.Label("Max Spectra"),
                dcc.Dropdown(
                    id="spectra-dropdown",
                    options=[{"label": str(v), "value": v} for v in max_spectra_values],
                    value=max_spectra_values[3],
                    clearable=False
                ),
            ], md=3),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Checklist(
                    options=[
                        {"label": "Training", "value": "Training"},
                        {"label": "Known Chemicals", "value": "Known Chemicals"},
                        {"label": "Novel Chemicals", "value": "Novel Chemicals"},
                    ],
                    value=["Training", "Known Chemicals", "Novel Chemicals"],
                    id="dataset-types",
                    inline=True
                ),
            ], md=6),
            dbc.Col([
                dbc.Checklist(
                    options=[
                        {"label": "Color by SMILES", "value": "color_by_smiles"},
                        {"label": "Novel Chemicals Black", "value": "novel_chemicals_black"},
                    ],
                    value=["color_by_smiles"],
                    id="color-mode-boxes",
                    inline=True
                ),
            ], md=6),
        ], className="mb-3"),
    ]),
    dcc.Loading(dcc.Graph(id='pca-graph'), type="circle"),
], fluid=True)

@app.callback(
    Output('pca-graph', 'figure'),
    Input('smiles-dropdown', 'value'),
    Input('spectra-dropdown', 'value'),
    Input('dataset-types', 'value'),
    Input('color-mode-boxes', 'value')
)
def update_pca_plot(max_smiles, max_spectra, selected_types, color_modes):
    include_training = "Training" in selected_types
    include_known_chemicals = "Known Chemicals" in selected_types
    include_novel_chemicals = "Novel Chemicals" in selected_types
    color_by_smiles = "color_by_smiles" in color_modes
    novel_chemicals_black = "novel_chemicals_black" in color_modes
    return make_pca_plot(
        max_smiles=max_smiles,
        max_spectra=max_spectra,
        color_by_smiles=color_by_smiles,
        novel_chemicals_black=novel_chemicals_black,
        include_novel_chemicals=include_novel_chemicals,
        include_known_chemicals=include_known_chemicals,
        include_training=include_training,
        random_seed=42
    )

if __name__ == '__main__':
    app.run(debug=True, port=8050)