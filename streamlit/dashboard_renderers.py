# dashboard_renderers.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle # For loading temporal data

# Import plotting libraries - these are needed for the _dashboard functions
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


# --- Dashboard Plotting Functions (Copied from the complete main.py provided earlier) ---
# These functions take dataframes and parameters and use st.pyplot or st.plotly_chart

def display_topic_words_dashboard(topic_id_sel, topics_df_local):
    if topics_df_local is None or topics_df_local.empty: st.warning("Topic words data not available."); return
    topic_words_data = topics_df_local[topics_df_local['topic_id'] == topic_id_sel]
    if topic_words_data.empty: st.warning(f"No words for Topic {topic_id_sel}."); return
    word_dict_cloud = dict(zip(topic_words_data['word'], topic_words_data['weight']))
    if not word_dict_cloud: st.warning(f"Word dictionary empty for Topic {topic_id_sel}."); return
    try:
        wc = WordCloud(width=800, height=300, background_color='white', colormap='viridis', max_words=30).generate_from_frequencies(word_dict_cloud)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4)); ax_wc.imshow(wc, interpolation='bilinear'); ax_wc.axis('off'); ax_wc.set_title(f'Word Cloud for Topic {topic_id_sel}')
        st.pyplot(fig_wc); plt.close(fig_wc)
    except Exception as e: st.error(f"Word cloud error: {e}")
    try:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 4)); sns.barplot(x='weight', y='word', data=topic_words_data.sort_values('weight', ascending=False).head(15), ax=ax_bar, palette="viridis")
        ax_bar.set_title(f'Top 15 Words for Topic {topic_id_sel}'); plt.tight_layout(); st.pyplot(fig_bar); plt.close(fig_bar)
    except Exception as e: st.error(f"Bar chart error: {e}")

def plot_user_topics_dashboard(user_id_str, user_topic_df_local, suspicious_df_local, num_topics_dash):
    if user_topic_df_local is None or user_topic_df_local.empty: return
    if user_id_str not in user_topic_df_local['user_id'].astype(str).values: st.warning(f"User ID '{user_id_str}' not found."); return
    user_data = user_topic_df_local[user_topic_df_local['user_id'].astype(str) == user_id_str].iloc[0]
    topic_props = [user_data.get(f'Topic_{i}', 0) for i in range(num_topics_dash)]
    threshold = 0.03; significant_topics = [{'id': i, 'prop': p} for i, p in enumerate(topic_props) if p > threshold]
    other_prop = sum(p for i, p in enumerate(topic_props) if p <= threshold)
    significant_topics.sort(key=lambda x: x['prop'], reverse=True)
    labels = [f'Topic {item["id"]} ({item["prop"]:.2f})' for item in significant_topics]; values = [item['prop'] for item in significant_topics]
    if other_prop > 0.001: labels.append(f'Other ({other_prop:.2f})'); values.append(other_prop)
    if not values: st.info(f"User {user_id_str} has no significant topic distribution."); return
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='percent', hoverinfo='label+percent', hole=0.3)])
    fig_pie.update_layout(title_text=f'Topic Distribution for User {user_id_str}', height=400, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown(f"**Metrics for User: {user_id_str}**"); st.markdown(f"- Post count: `{user_data.get('post_count', 'N/A')}`")
    st.markdown(f"- Gini: `{user_data.get('gini_coefficient', float('nan')):.4f}` | Shannon: `{user_data.get('shannon_entropy', float('nan')):.4f}` | Top-1 Ratio: `{user_data.get('top1_ratio', float('nan')):.4f}`")
    if suspicious_df_local is not None and user_id_str in suspicious_df_local['user_id'].astype(str).values:
        susp_data = suspicious_df_local[suspicious_df_local['user_id'].astype(str) == user_id_str].iloc[0]
        if susp_data.get('suspicious', False):
            st.warning("⚠️ This user has been flagged as suspicious.")
            reasons = [r for r, c in [("high posting frequency", 'suspicious_post_freq'), ("narrow topic focus", 'suspicious_narrowness'), ("high duplicate content", 'suspicious_duplicates')] if susp_data.get(c, False)]
            if reasons: st.write(f"   Reasons: {', '.join(reasons)}")

def plot_topic_concentration_dashboard(metric_col, log_scale_flag, user_topic_df_local, suspicious_df_local):
    if user_topic_df_local is None or user_topic_df_local.empty: return
    if metric_col not in user_topic_df_local.columns: st.error(f"Metric '{metric_col}' not in data."); return
    fig_conc = px.scatter(user_topic_df_local, x='post_count', y=metric_col, hover_data=['user_id', 'post_count', 'gini_coefficient', 'shannon_entropy', 'top1_ratio'], color='dominant_topic', size='post_count', size_max=25, opacity=0.7, log_x=log_scale_flag, title=f'User Topic Concentration ({metric_col.replace("_", " ").title()}) vs Post Frequency')
    mean_val_conc = user_topic_df_local[metric_col].mean(); fig_conc.add_hline(y=mean_val_conc, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_val_conc:.3f}", annotation_position="bottom right")
    if suspicious_df_local is not None and 'suspicious' in suspicious_df_local.columns:
        susp_users_list = suspicious_df_local[suspicious_df_local['suspicious']]['user_id'].astype(str).tolist()
        susp_plot_df = user_topic_df_local[user_topic_df_local['user_id'].astype(str).isin(susp_users_list)]
        if not susp_plot_df.empty: fig_conc.add_trace(go.Scatter(x=susp_plot_df['post_count'], y=susp_plot_df[metric_col], mode='markers', marker=dict(color='rgba(0,0,0,0)', size=15, line=dict(color='red', width=3), symbol='circle-open'), name='Suspicious Outline', text=susp_plot_df['user_id'], hoverinfo='text'))
    fig_conc.update_layout(xaxis_title="Number of Posts", yaxis_title=f"{metric_col.replace('_', ' ').title()}", height=500, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig_conc, use_container_width=True)

def plot_suspicious_comparison_dashboard(metric_col_susp, user_topic_df_local, suspicious_df_local):
    if user_topic_df_local is None or suspicious_df_local is None: return
    user_topic_df_local['user_id'] = user_topic_df_local['user_id'].astype(str); suspicious_df_local['user_id'] = suspicious_df_local['user_id'].astype(str)
    merged_df = user_topic_df_local.merge(suspicious_df_local[['user_id', 'suspicious', 'duplicate_post_ratio']], on='user_id', how='left')
    merged_df['suspicious'] = merged_df['suspicious'].fillna(False); merged_df['user_type'] = merged_df['suspicious'].map({True: 'Suspicious', False: 'Normal'})
    if metric_col_susp == 'duplicate_post_ratio' and 'duplicate_post_ratio' not in merged_df.columns: st.error("'duplicate_post_ratio' column not found."); return
    if metric_col_susp not in merged_df.columns: st.error(f"Metric column '{metric_col_susp}' not found."); return
    fig_violin = px.violin(merged_df, x='user_type', y=metric_col_susp, color='user_type', box=True, points='all', hover_data=['user_id', 'post_count'], title=f'Distribution of {metric_col_susp.replace("_", " ").title()}: Suspicious vs Normal Users')
    fig_violin.update_layout(height=450, margin=dict(t=50, b=20, l=20, r=20)); st.plotly_chart(fig_violin, use_container_width=True)

def plot_topic_similarity_heatmap_dashboard(user_topic_df_local, num_topics_dash):
    if user_topic_df_local is None: return
    topic_cols_sim = [f'Topic_{i}' for i in range(num_topics_dash)]
    for col in topic_cols_sim:
        if col not in user_topic_df_local.columns: user_topic_df_local[col] = 0 
    topic_vectors_sim = user_topic_df_local[topic_cols_sim].T.values 
    if topic_vectors_sim.shape[0] == 0: st.warning("No topics for similarity matrix."); return
    similarity_matrix = cosine_similarity(topic_vectors_sim) 
    fig_heatmap = px.imshow(similarity_matrix, labels=dict(x="Topic", y="Topic", color="Similarity"), x=[f'T{i}' for i in range(num_topics_dash)], y=[f'T{i}' for i in range(num_topics_dash)], color_continuous_scale='Viridis', aspect="auto")
    fig_heatmap.update_layout(title='Topic Similarity Matrix (Cosine Similarity)', height=500, margin=dict(t=50, b=20, l=20, r=20)); st.plotly_chart(fig_heatmap, use_container_width=True)

def plot_topic_similarity_network_dashboard(user_topic_df_local, num_topics_dash, min_similarity_threshold):
    if user_topic_df_local is None: st.warning("User topic data not available for network plot."); return
    topic_cols_net = [f'Topic_{i}' for i in range(num_topics_dash)]; 
    for col in topic_cols_net:
        if col not in user_topic_df_local.columns: user_topic_df_local[col] = 0
    topic_vectors_net = user_topic_df_local[topic_cols_net].T.values
    if topic_vectors_net.shape[0] == 0: st.warning("No topics for network plot."); return
    similarity_matrix_net = cosine_similarity(topic_vectors_net)
    G = nx.Graph(); 
    for i in range(num_topics_dash): G.add_node(i, label=f"T{i}")
    for i in range(num_topics_dash):
        for j in range(i + 1, num_topics_dash):
            if similarity_matrix_net[i, j] >= min_similarity_threshold: G.add_edge(i, j, weight=similarity_matrix_net[i, j])
    if not G.edges(): st.info("No topic pairs meet the similarity threshold to draw a network."); return
    pos = nx.spring_layout(G, seed=42, k=0.7/np.sqrt(G.number_of_nodes()) if G.number_of_nodes() > 1 else 0.7) 
    edge_x, edge_y = [], []; 
    for edge in G.edges(): x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]; edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_text, node_size_val = [], [], [], []
    for node in G.nodes(): 
        x, y = pos[node]; node_x.append(x); node_y.append(y); node_text.append(G.nodes[node]['label']); node_size_val.append(15 + G.degree(node) * 3)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center", hoverinfo='text', 
                            marker=dict(showscale=True, colorscale='YlGnBu', size=node_size_val, 
                                        colorbar=dict(thickness=15, title='Node Degree', xanchor='left', titleside='right'), 
                                        line_width=2, color=[G.degree(node) for node in G.nodes()]))
    fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='Topic Similarity Network', titlefont_size=16, showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=500))
    st.plotly_chart(fig_net, use_container_width=True)

def plot_user_clustering_dashboard(n_clusters_viz, show_stats_flag, show_heatmap_flag, user_topic_df_local, suspicious_df_local, num_topics_dash):
    if user_topic_df_local is None: return
    topic_cols_cluster = [f'Topic_{i}' for i in range(num_topics_dash)]; 
    for col in topic_cols_cluster:
        if col not in user_topic_df_local.columns: user_topic_df_local[col] = 0
    X_cluster = user_topic_df_local[topic_cols_cluster].fillna(0).values
    if X_cluster.shape[0] < 2 or X_cluster.shape[0] < n_clusters_viz: st.warning(f"Not enough users ({X_cluster.shape[0]}) or too few for {n_clusters_viz} clusters."); return
    perplexity_val = min(30.0, max(5.0, X_cluster.shape[0] - 1.0)) 
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=300, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_cluster)
    except Exception as e: st.error(f"t-SNE error: {e}"); return
    current_n_clusters = n_clusters_viz
    if X_cluster.shape[0] < n_clusters_viz : current_n_clusters = max(1, X_cluster.shape[0]); st.warning(f"Reduced clusters to {current_n_clusters}.")
    if current_n_clusters <= 1: st.error("Cannot perform clustering with less than 2 clusters."); return
    kmeans = KMeans(n_clusters=current_n_clusters, random_state=42, n_init='auto'); clusters = kmeans.fit_predict(X_cluster)
    plot_df = pd.DataFrame({'x': X_tsne[:, 0], 'y': X_tsne[:, 1], 'user_id': user_topic_df_local['user_id'], 'cluster': clusters.astype(str), 'post_count': user_topic_df_local['post_count'], 'gini': user_topic_df_local['gini_coefficient']})
    fig_scatter = px.scatter(plot_df, x='x', y='y', color='cluster', size='post_count', hover_data=['user_id', 'gini', 'post_count'], size_max=20, opacity=0.7, title=f'User Clustering (K-means, {current_n_clusters} clusters) with t-SNE')
    if suspicious_df_local is not None and 'suspicious' in suspicious_df_local.columns:
        susp_ids = suspicious_df_local[suspicious_df_local['suspicious']]['user_id'].astype(str).tolist()
        susp_plot_df_points = plot_df[plot_df['user_id'].astype(str).isin(susp_ids)]
        if not susp_plot_df_points.empty: fig_scatter.add_trace(go.Scatter(x=susp_plot_df_points['x'], y=susp_plot_df_points['y'], mode='markers', marker=dict(color='rgba(0,0,0,0)', size=18, line=dict(color='red', width=3), symbol='circle-open'), name='Suspicious Outline', hoverinfo='text', text=susp_plot_df_points['user_id']))
    fig_scatter.update_layout(xaxis_title="t-SNE Dimension 1", yaxis_title="t-SNE Dimension 2", height=600, margin=dict(t=50,b=20,l=20,r=20)); st.plotly_chart(fig_scatter, use_container_width=True)
    if show_stats_flag:
        cluster_stats = plot_df.groupby('cluster').agg(User_Count=('user_id', 'count'), Avg_Gini=('gini', 'mean'), Avg_Post_Count=('post_count', 'mean')).reset_index()
        st.write("Cluster Statistics:"); st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
        if show_heatmap_flag:
            st.write("Dominant Topics per Cluster (Mean Weights):")
            cluster_topic_means = []
            for cid_str in sorted(plot_df['cluster'].unique()):
                user_ids_in_cluster = plot_df[plot_df['cluster'] == cid_str]['user_id']
                cluster_data = user_topic_df_local[user_topic_df_local['user_id'].isin(user_ids_in_cluster)]
                means = [cluster_data.get(f'Topic_{i}', pd.Series(dtype=float)).mean() for i in range(num_topics_dash)]
                cluster_topic_means.append(means)
            heatmap_df = pd.DataFrame(cluster_topic_means, columns=[f'T{i}' for i in range(num_topics_dash)], index=[f'Cluster {i}' for i in sorted(plot_df['cluster'].unique())])
            fig_heatmap_cluster = px.imshow(heatmap_df.fillna(0), labels=dict(x="Topic", y="Cluster", color="Mean Weight"), color_continuous_scale='Viridis', aspect="auto")
            fig_heatmap_cluster.update_layout(title='Cluster-Topic Distribution Heatmap', height=max(300, 40 * current_n_clusters), margin=dict(t=50,b=20,l=20,r=20)); st.plotly_chart(fig_heatmap_cluster, use_container_width=True)

def plot_temporal_topics_dashboard(user_id_str, show_individual_flag, temporal_data_dict, num_topics_dash):
    if not temporal_data_dict or user_id_str not in temporal_data_dict: st.info(f"No temporal data for user {user_id_str}."); return
    user_temporal_df = temporal_data_dict[user_id_str]
    if not isinstance(user_temporal_df, pd.DataFrame) or user_temporal_df.empty: st.info(f"Temporal data for {user_id_str} empty/invalid."); return
    try: user_temporal_df.index = pd.to_datetime(user_temporal_df.index)
    except: st.warning("Could not convert temporal index to datetime."); 
    top_n = 5; avg_dist = user_temporal_df.mean()
    existing_cols = [col for col in [f'Topic_{i}' for i in range(num_topics_dash)] if col in avg_dist.index]
    if not existing_cols: st.warning("No topic columns in temporal data."); return
    top_topics = avg_dist[existing_cols].nlargest(top_n).index.tolist()
    plot_df = user_temporal_df.copy()
    other_cols = [col for col in existing_cols if col not in top_topics]
    if other_cols: plot_df['Other_Topics'] = user_temporal_df[other_cols].sum(axis=1)
    fig_stack = go.Figure()
    for topic in top_topics:
        if topic in plot_df.columns: fig_stack.add_trace(go.Scatter(x=plot_df.index, y=plot_df[topic], mode='lines', stackgroup='one', name=topic))
    if 'Other_Topics' in plot_df.columns: fig_stack.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Other_Topics'], mode='lines', stackgroup='one', name='Other Topics'))
    fig_stack.update_layout(title=f'Topic Evolution Over Time for User {user_id_str} (Stacked)', xaxis_title='Time', yaxis_title='Topic Proportion', hovermode='x unified', height=450, margin=dict(t=50,b=20,l=20,r=20))
    st.plotly_chart(fig_stack, use_container_width=True)
    if show_individual_flag:
        fig_ind = go.Figure()
        for topic in top_topics:
            if topic in user_temporal_df.columns: fig_ind.add_trace(go.Scatter(x=user_temporal_df.index, y=user_temporal_df[topic], mode='lines+markers', name=topic))
        fig_ind.update_layout(title=f'Individual Top Topic Evolution for User {user_id_str}', xaxis_title='Time', yaxis_title='Topic Proportion', height=450, margin=dict(t=50,b=20,l=20,r=20))
        st.plotly_chart(fig_ind, use_container_width=True)


# --- Main Function to Render the Detailed Dashboard ---
def render_full_dashboard(results_dir_path_str):
    st.header("🔬 Detailed Topic Analysis Dashboard")
    st.markdown("Explore the results of the LDA topic modeling pipeline.")
    results_dir = Path(results_dir_path_str)

    # --- Load Data ---
    user_topic_df, suspicious_df, topics_df, temporal_topic_data_loaded = None, None, None, None
    # Get num_topics used for the run from session state (set by lda_pipeline_orchestrator)
    num_topics_dash = st.session_state.get("lda_num_topics_run", 10) 

    user_topic_df_path = results_dir / "user_topic_data.csv"
    if user_topic_df_path.exists(): 
        user_topic_df = pd.read_csv(user_topic_df_path)
    else: 
        st.error(f"Required file not found: {user_topic_df_path}")
        return # Stop dashboard rendering if essential data is missing

    suspicious_df_path = results_dir / "suspicious_users.csv"
    if suspicious_df_path.exists(): 
        suspicious_df = pd.read_csv(suspicious_df_path)
    # else: st.warning("suspicious_users.csv not found. Some dashboard features might be limited.")

    topics_df_path = results_dir / "topic_words.csv"
    if topics_df_path.exists(): 
        topics_df = pd.read_csv(topics_df_path)
    # else: st.warning("topic_words.csv not found. Topic word visualizations will be unavailable.")
    
    temporal_data_pickle_path = results_dir / "temporal_topic_data.pkl" 
    if temporal_data_pickle_path.exists():
        try:
            with open(temporal_data_pickle_path, 'rb') as f_pkl: 
                temporal_topic_data_loaded = pickle.load(f_pkl)
        except Exception as e_pkl: 
            st.warning(f"Could not load temporal_topic_data.pkl: {e_pkl}")
    
    # --- Dashboard Tabs ---
    tab_titles = ["Topic Words", "User Distributions", "Topic Concentration", 
                  "Suspicious Users", "Temporal Analysis", "Clustering & Similarity"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

    with tab1: # Topic Words
        st.subheader("Topic Word Distributions")
        if topics_df is not None and not topics_df.empty:
            # Ensure num_topics_dash is valid for the slider
            max_slider_val = max(0, num_topics_dash - 1)
            if max_slider_val == 0 and num_topics_dash > 0: # If only 1 topic (index 0)
                 selected_topic_id_dash = 0
                 st.write(f"Displaying Topic {selected_topic_id_dash} (Only one topic model or data limited)")
            elif num_topics_dash > 0:
                 selected_topic_id_dash = st.slider("Select Topic ID", 0, max_slider_val, 0, key="dash_topic_slider")
            else:
                st.info("No topics available to display words for.")
                selected_topic_id_dash = -1 # Indicates no valid topic
            
            if selected_topic_id_dash >= 0:
                 display_topic_words_dashboard(selected_topic_id_dash, topics_df)
        else: 
            st.info("Topic words data (topic_words.csv) not loaded or is empty.")

    with tab2: # User Distributions
        st.subheader("User Topic Distribution")
        if user_topic_df is not None and not user_topic_df.empty:
            user_list_dash = sorted(user_topic_df['user_id'].astype(str).unique())
            if user_list_dash:
                selected_user_dash = st.selectbox("Select User", user_list_dash, index=0, key="dash_user_select")
                plot_user_topics_dashboard(selected_user_dash, user_topic_df, suspicious_df, num_topics_dash)
            else: 
                st.info("No users found in the data.")
        else: 
            st.info("User topic data not loaded.")
            
    with tab3: # Topic Concentration
        st.subheader("Topic Concentration vs. Post Frequency")
        if user_topic_df is not None and not user_topic_df.empty:
            metric_options = {'Gini Coefficient': 'gini_coefficient', 
                              'Shannon Entropy': 'shannon_entropy', 
                              'Top-1 Topic Ratio': 'top1_ratio'}
            selected_metric_display = st.selectbox("Select Metric", list(metric_options.keys()), key="dash_conc_metric_selector")
            selected_metric_col = metric_options[selected_metric_display]
            log_scale_ui = st.checkbox("Use Log Scale for Post Count", False, key="dash_conc_log_checkbox")
            plot_topic_concentration_dashboard(selected_metric_col, log_scale_ui, user_topic_df, suspicious_df)
        else: 
            st.info("User topic data not loaded.")

    with tab4: # Suspicious Users
        st.subheader("Suspicious Users Analysis")
        if suspicious_df is not None and user_topic_df is not None and 'suspicious' in suspicious_df.columns:
            if suspicious_df['suspicious'].sum() > 0:
                susp_users_display = suspicious_df[suspicious_df['suspicious']]
                st.write("Top 10 Flagged Suspicious Users (if any):")
                st.dataframe(susp_users_display[['user_id', 'post_count', 'gini_coefficient', 'top1_ratio', 'duplicate_post_ratio']].head(10).style.background_gradient(cmap='Reds'))
                
                metric_options_susp_comp = {'Gini Coefficient': 'gini_coefficient', 
                                          'Duplicate Post Ratio': 'duplicate_post_ratio',
                                          'Shannon Entropy': 'shannon_entropy',
                                          'Top-1 Topic Ratio': 'top1_ratio'}
                selected_metric_susp_comp_disp = st.selectbox("Select Metric for Comparison (Violin Plot)", list(metric_options_susp_comp.keys()), key="dash_susp_compare_metric_selector")
                plot_suspicious_comparison_dashboard(metric_options_susp_comp[selected_metric_susp_comp_disp], user_topic_df, suspicious_df)
            else: 
                st.info("No users were flagged as suspicious in this run.")
        else: 
            st.info("Suspicious user data or user topic data not available/loaded.")

    with tab5: # Temporal Analysis
        st.subheader("Temporal Topic Analysis")
        if temporal_topic_data_loaded:
            user_list_temporal = sorted([str(k) for k in temporal_topic_data_loaded.keys()])
            if user_list_temporal:
                selected_user_temporal = st.selectbox("Select User for Temporal Plot", user_list_temporal, key="dash_temp_user_selector")
                show_individual_lines_temporal = st.checkbox("Show Individual Top Topic Lines", False, key="dash_temp_individual_checkbox")
                plot_temporal_topics_dashboard(selected_user_temporal, show_individual_lines_temporal, temporal_topic_data_loaded, num_topics_dash)
            else: 
                st.info("No users found with temporal data.")
        else:
            st.warning("Temporal data file (temporal_topic_data.pkl) not found or could not be loaded. "
                       "Ensure your UserTopicModelingSystem saves this file in its output directory if this feature is desired.")
    
    with tab6: # Clustering & Similarity
        st.subheader("Topic Similarity & User Clustering")
        sim_col, clust_col = st.columns(2)
        with sim_col:
            st.markdown("#### Topic Similarity")
            if user_topic_df is not None: 
                plot_topic_similarity_heatmap_dashboard(user_topic_df, num_topics_dash)
                min_sim_for_network = st.slider("Min Similarity for Network Edge", 0.0, 1.0, 0.3, 0.05, key="dash_topic_net_sim_slider_main")
                plot_topic_similarity_network_dashboard(user_topic_df, num_topics_dash, min_sim_for_network)
            else: st.info("User topic data not loaded.")
        
        with clust_col:
            st.markdown("#### User Clustering (t-SNE)")
            if user_topic_df is not None:
                num_clusters_for_dash = st.slider("Number of User Clusters (K-Means)", 2, 15, 5, key="dash_user_clust_num_main_slider")
                show_stats_for_clusters = st.checkbox("Show Cluster Stats", True, key="dash_user_clust_stats_main_cb")
                show_heatmap_for_clusters = st.checkbox("Show Cluster-Topic Heatmap", True, key="dash_user_clust_hm_main_cb")
                plot_user_clustering_dashboard(num_clusters_for_dash, show_stats_for_clusters, show_heatmap_for_clusters, user_topic_df, suspicious_df, num_topics_dash)
            else: st.info("User topic data not loaded.")

    st.markdown("---")
    if st.button("⬅️ Back to Main Setup & Results", key="back_to_main_from_dashboard_btn_final"):
        st.session_state.viewing_dashboard = False
        st.session_state.show_detailed_dashboard_button = True
        st.experimental_rerun()