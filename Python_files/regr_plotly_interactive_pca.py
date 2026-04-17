import dash
from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.colors as pc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---- Sourcing of data files
bin_size = 0.5
threshold = 0.5

# Explicitly set the files to use
known_file = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/plotly_dataframes/plotly_df_regr.parquet"
novel_file = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/plotly_dataframes/plotly_df_regr_super_test.parquet"
chemnet_file = "/home/dlipsey/MITLincolnLabs/MIT_LL_data/plotly_dataframes/df6_chemnet.parquet"
smiles_col = "SMILES_spectra"

# --- LD50 (mg/kg) → tox_level bins
def assign_tox_level(response):
    if response <= 5:
        return 0
    elif response <= 50:
        return 1
    elif response <= 500:
        return 2
    elif response <= 5000:
        return 3
    else:
        return 4

def sample_spectra_per_chemical(df, smiles_col, max_per_smiles, seed):
    selected_rows = []
    for smile in df[smiles_col].unique():
        smile_rows = df[df[smiles_col] == smile]
        if len(smile_rows) > max_per_smiles:
            selected_rows.extend(smile_rows.sample(max_per_smiles, random_state=seed).index)
        else:
            selected_rows.extend(smile_rows.index)
    return df.loc[selected_rows]

def balanced_smiles_selection(df, smiles_col, tox_col, max_smiles, seed):
    # df: must have unique [smiles_col, tox_col] rows!
    smiles_tox = df[[smiles_col, tox_col]].drop_duplicates()
    tox_counts = smiles_tox[tox_col].value_counts()
    n_classes = tox_counts.shape[0]
    min_per_class = max_smiles // n_classes if n_classes > 0 else 0
    remaining = max_smiles
    selected = []
    rng = np.random.default_rng(seed)
    for tox_level, count in tox_counts.sort_index().items():
        available = smiles_tox[smiles_tox[tox_col] == tox_level][smiles_col].values
        n_select = min(min_per_class, count, remaining)
        if n_select > 0:
            selected.extend(rng.choice(available, size=n_select, replace=False))
            remaining -= n_select
    # If there's quota, fill from everything else left
    if remaining > 0:
        remaining_smiles = smiles_tox[~smiles_tox[smiles_col].isin(selected)][smiles_col].values
        n_available = len(remaining_smiles)
        if n_available > 0:
            n_pick = min(remaining, n_available)
            selected.extend(rng.choice(remaining_smiles, size=n_pick, replace=False))
            remaining -= n_pick
    return df[df[smiles_col].isin(selected)]

def load_data(file_path, smiles_col, max_smiles, max_spectra, filter_train=None, required_smiles=None, seed=None, balance_classes=False):
    try:
        df = pd.read_parquet(file_path)
        if filter_train is not None and 'train' in df.columns:
            df = df[df['train'] == filter_train]
        embedding_cols = [col for col in df.columns if col.startswith('cond_emb_')
                          and 'morgan' not in col.lower() and 'filtered' not in col.lower()]
        if not embedding_cols or len(df) == 0:
            return None, None, None, None, set(), None

        # Assign regression tox level to each SMILES based on median Response for balancing.
        if balance_classes and 'Response' in df.columns:
            smile2response = df.groupby(smiles_col)['Response'].median()
            smile2tox = smile2response.apply(assign_tox_level)
            df = df.merge(smile2tox.rename("bal_tox_level"), left_on=smiles_col, right_index=True)
        else:
            df["bal_tox_level"] = None

        if required_smiles is not None and len(required_smiles) > 0:
            df = df[df[smiles_col].isin(required_smiles)]
        else:
            unique_smiles = pd.unique(df[smiles_col])
            if len(unique_smiles) > max_smiles:
                if balance_classes and 'bal_tox_level' in df.columns and df['bal_tox_level'].notna().any():
                    df = balanced_smiles_selection(df, smiles_col, 'bal_tox_level', max_smiles, seed)
                else:
                    if seed is not None:
                        np.random.seed(seed)
                    selected_smiles_subset = np.random.choice(unique_smiles, max_smiles, replace=False)
                    df = df[df[smiles_col].isin(selected_smiles_subset)]

        df = sample_spectra_per_chemical(df, smiles_col, max_spectra, seed)
        embeddings = df[embedding_cols].values
        smiles = df[smiles_col].values
        true_vals = df['Response'].values if 'Response' in df.columns else [None] * len(df)
        pred_vals = df['cond_tox_pred'].values if 'cond_tox_pred' in df.columns else [None] * len(df)
        return embeddings, smiles, true_vals, pred_vals, set(smiles), df.get('bal_tox_level', pd.Series([None]*len(df)))
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, set(), None

def load_chemnet_data(file_path, selected_smiles, max_spectra, smiles_col, seed, smiles_to_true_ld50_map):
    try:
        df = pd.read_parquet(file_path)
        embedding_cols = [col for col in df.columns if col.startswith('Embedding Float ')]
        embedding_cols.sort(key=lambda x: int(x.split()[-1]))
        df = df[df[smiles_col].isin(selected_smiles)]
        df = sample_spectra_per_chemical(df, smiles_col, max_spectra, seed)
        if not embedding_cols or len(df) == 0:
            return None, None, None, None
        embeddings = df[embedding_cols].values
        smiles = df[smiles_col].values
        true_vals = [smiles_to_true_ld50_map.get(smile, None) for smile in smiles]
        pred_vals = [None] * len(df)
        return embeddings, smiles, true_vals, pred_vals
    except Exception as e:
        print(f"Error loading ChemNet data: {e}")
        return None, None, None, None

def make_pca_plot(
    max_smiles=20,
    max_spectra=10,
    color_by_smiles=True,
    novel_chemicals_black=False,
    include_novel_chemicals=True,
    include_known_chemicals=True,
    include_true_chemnet=True,
    include_training=True,
    random_seed=42,
    balance_classes=False
):
    known_emb, known_smiles, known_true, known_pred, known_smiles_set, known_tox_levels = None, None, None, None, set(), None
    if include_known_chemicals:
        known_emb, known_smiles, known_true, known_pred, known_smiles_set, known_tox_levels = \
            load_data(known_file, smiles_col, max_smiles, max_spectra, filter_train=0, seed=random_seed, balance_classes=balance_classes)
    train_emb, train_smiles, train_true, train_pred, _, train_tox_levels = None, None, None, None, set(), None
    if include_training:
        train_emb, train_smiles, train_true, train_pred, _, train_tox_levels = \
            load_data(known_file, smiles_col, max_smiles, max_spectra, filter_train=1, required_smiles=known_smiles_set if include_known_chemicals else None, seed=random_seed, balance_classes=balance_classes)
    novel_emb, novel_smiles, novel_true, novel_pred, _, novel_tox_levels = None, None, None, None, set(), None
    if include_novel_chemicals:
        novel_emb, novel_smiles, novel_true, novel_pred, _, novel_tox_levels = \
            load_data(novel_file, smiles_col, max_smiles, max_spectra, seed=random_seed, balance_classes=balance_classes)
    smiles_to_true_ld50_map = {}
    for arr_smiles, arr_true in [(known_smiles, known_true), (train_smiles, train_true), (novel_smiles, novel_true)]:
        if arr_smiles is not None:
            for smile, tcls in zip(arr_smiles, arr_true):
                if tcls is not None:
                    smiles_to_true_ld50_map[smile] = tcls
    selected_smiles_all = set()
    for arr in [known_smiles, train_smiles, novel_smiles]:
        if arr is not None:
            selected_smiles_all.update(arr)
    chemnet_emb, chemnet_smiles, chemnet_true, chemnet_pred = None, None, None, None
    if include_true_chemnet:
        chemnet_emb, chemnet_smiles, chemnet_true, chemnet_pred = \
            load_chemnet_data(chemnet_file, selected_smiles_all, max_spectra, smiles_col, random_seed, smiles_to_true_ld50_map)
    trace_types = []
    if include_training and train_emb is not None:
        trace_types.append(('Training', train_emb, train_smiles, train_true, train_pred))
    if include_known_chemicals and known_emb is not None:
        trace_types.append(('Known Chemicals', known_emb, known_smiles, known_true, known_pred))
    if include_novel_chemicals and novel_emb is not None:
        trace_types.append(('Novel Chemicals', novel_emb, novel_smiles, novel_true, novel_pred))
    if include_true_chemnet and chemnet_emb is not None:
        trace_types.append(('True ChemNet', chemnet_emb, chemnet_smiles, chemnet_true, chemnet_pred))
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
    for dtype in ['Training','Known Chemicals','Novel Chemicals','True ChemNet']:
        mask = dataset_types_array==dtype
        if not mask.any(): continue
        if novel_chemicals_black:
            color_arr = []
            for idx, flag in enumerate(mask):
                if not flag: continue
                smile = smiles_array[idx]
                if dtype == "Novel Chemicals":
                    color_arr.append("black")
                elif dtype == "True ChemNet" and smile in current_novel_smiles_set:
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
            try:
                true_ld50_txt = f"{float(v_true):.3f}" if v_true is not None and not pd.isna(v_true) else "N/A"
            except Exception:
                true_ld50_txt = "N/A"
            if dtype == "True ChemNet":
                txt = f"SMILES: {smile}<br>True LD50: {true_ld50_txt}<br>Predicted LD50: N/A"
            else:
                try:
                    pred_ld50 = np.exp(v_pred) if v_pred is not None and not pd.isna(v_pred) else None
                    pred_ld50_txt = f"{float(pred_ld50):.3f}" if pred_ld50 is not None else "N/A"
                except:
                    pred_ld50_txt = 'N/A'
                txt = f"SMILES: {smile}<br>True LD50: {true_ld50_txt}<br>Predicted LD50: {pred_ld50_txt}"
            subset_hover.append(txt)
        marker_map = {
            'Training': 'triangle-up',
            'Known Chemicals': 'square',
            'Novel Chemicals': 'circle',
            'True ChemNet': 'x'
        }
        marker = marker_map[dtype]
        alpha = 1.0 if dtype=="True ChemNet" else 0.7
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
            text=f'2D PCA of Regression Encoder Embeddings (bin={bin_size}, thresh={threshold})',
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

max_smiles_values = [5, 10, 25, 50, 100, 200]
max_spectra_values = [1, 5, 10, 25, 50, 100, 200]

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
                        {"label": "True ChemNet", "value": "True ChemNet"},
                    ],
                    value=["Training", "Known Chemicals", "Novel Chemicals", "True ChemNet"],
                    id="dataset-types",
                    inline=True
                ),
            ], md=5),
            dbc.Col([
                dbc.Checklist(
                    options=[
                        {"label": "Color by SMILES", "value": "color_by_smiles"},
                        {"label": "Novel Chemicals Black", "value": "novel_chemicals_black"},
                        {"label": "Balance Classes", "value": "balance_classes"},
                    ],
                    value=["color_by_smiles"],
                    id="color-mode-boxes",
                    inline=True
                ),
            ], md=7),
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
    include_true_chemnet = "True ChemNet" in selected_types
    color_by_smiles = "color_by_smiles" in color_modes
    novel_chemicals_black = "novel_chemicals_black" in color_modes
    balance_classes = "balance_classes" in color_modes
    return make_pca_plot(
        max_smiles=max_smiles,
        max_spectra=max_spectra,
        color_by_smiles=color_by_smiles,
        novel_chemicals_black=novel_chemicals_black,
        include_novel_chemicals=include_novel_chemicals,
        include_known_chemicals=include_known_chemicals,
        include_true_chemnet=include_true_chemnet,
        include_training=include_training,
        random_seed=42,
        balance_classes=balance_classes
    )

if __name__ == '__main__':
    app.run(debug=True, port=8051)