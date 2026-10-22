import streamlit as st
import pandas as pd
import numpy as np
import io
import json

# --- VERIFICATION DES MODULES IA ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

st.set_page_config(page_title="Convertisseur Cashmag Pro", page_icon="🛒", layout="centered")

st.title("🛒 Convertisseur Cashmag Universel")
st.write("Importez n'importe quel fichier article (Standard, Multi-pages, ou Mercuriale). L'outil s'adapte à tout.")

CASHMAG_COLS = [
    'RUBRIQUE', 'ABREGE PRODUIT', 'LIBELLE PRODUIT', 'REGROUPEMENT', 
    'CODE BARRE', 'CODE FOURNISSEUR', 'PRIX LIBRE (prix variable)', 
    'PRIX', 'PRIX 2 (A emporter)', 'PRIX 3 (Autre)', 'DESCRIPTION'
]

# --- FONCTIONS UTILITAIRES ---

def load_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        xls = pd.ExcelFile(uploaded_file)
        all_dfs = []
        if len(xls.sheet_names) == 1:
            return pd.read_excel(xls, sheet_name=0)
            
        for sheet_name in xls.sheet_names:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
            if not df_sheet.empty:
                all_dfs.append(df_sheet)
        return pd.concat(all_dfs, ignore_index=True)

def enforce_cashmag_rules(df):
    df['RUBRIQUE'] = df['RUBRIQUE'].replace('', 'SANS RUBRIQUE')
    df['RUBRIQUE'] = df['RUBRIQUE'].fillna('SANS RUBRIQUE')
    
    df['LIBELLE PRODUIT'] = df['LIBELLE PRODUIT'].replace('', 'PRODUIT_SANS_NOM')
    df['LIBELLE PRODUIT'] = df['LIBELLE PRODUIT'].fillna("PRODUIT_SANS_NOM")
    
    counts = df.groupby('LIBELLE PRODUIT').cumcount()
    df['LIBELLE PRODUIT'] = df.apply(
        lambda row: str(row['LIBELLE PRODUIT']) if counts[row.name] == 0 else f"{row['LIBELLE PRODUIT']} ({counts[row.name] + 1})", 
        axis=1
    )
    df['ABREGE PRODUIT'] = df['LIBELLE PRODUIT']
    return df

def auto_format_mercuriale(df):
    df = df.dropna(axis=1, how='all')
    if len(df.columns) > 4:
        return df, False

    new_rows = []
    current_category = "SANS RUBRIQUE"
    text_col = df.columns[0]
    price_col = df.columns[-1]
    
    for index, row in df.iterrows():
        text_val = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
        price_val = row[price_col]
        
        if text_val and text_val.lower() not in ["nan", "tarifs", "prix"]:
            try:
                float_price = float(str(price_val).replace(',', '.'))
                if np.isnan(float_price): raise ValueError
                new_rows.append({"RUBRIQUE": current_category, "LIBELLE PRODUIT": text_val, "PRIX": float_price})
            except ValueError:
                current_category = text_val
                
    if new_rows:
        return pd.DataFrame(new_rows), True
    return df, False

def guess_column(target, available_columns):
    target = target.lower()
    for col in available_columns:
        col_lower = str(col).lower()
        if target == 'rubrique' and any(word in col_lower for word in ['famille', 'catégorie', 'rubrique', 'rayon']): return col
        if 'libelle' in target and any(word in col_lower for word in ['nom', 'libellé', 'article', 'désignation', 'produit']): return col
        if target == 'prix' and any(word in col_lower for word in ['prix', 'tarif', 'ttc', 'vente']): return col
        if 'barre' in target and any(word in col_lower for word in ['barre', 'ean', 'gencod', 'codebar']): return col
        if 'fournisseur' in target and any(word in col_lower for word in ['réf', 'ref', 'fournisseur']): return col
        if target == 'regroupement' and any(word in col_lower for word in ['tva', 'taxe', 'taux']): return col
    return "Aucune"

# --- INTERFACE UTILISATEUR ---

st.sidebar.title("⚙️ Paramètres Avancés")
use_ai = st.sidebar.toggle("🧠 Activer l'Intelligence Artificielle")
api_key = ""
ai_provider = "OpenAI"

if use_ai:
    ai_provider = st.sidebar.radio("Choisissez le moteur IA :", ["OpenAI", "Google Gemini"])
    
    if ai_provider == "OpenAI" and not OPENAI_AVAILABLE:
        st.sidebar.error("Module manquant. Tapez : pip install openai")
    elif ai_provider == "Google Gemini" and not GEMINI_AVAILABLE:
        st.sidebar.error("Module manquant. Tapez : pip install google-generativeai")
    else:
        api_key = st.sidebar.text_input(f"Clé API {ai_provider}", type="password")
        st.sidebar.info("L'IA permet d'analyser des fichiers chaotiques, de déduire automatiquement la TVA (Martinique) et de générer les rubriques manquantes.")

uploaded_file = st.file_uploader("📂 Glissez votre fichier ici (Excel/CSV)", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    df_raw = load_file(uploaded_file)
    df_final = pd.DataFrame(columns=CASHMAG_COLS)
    conversion_ready = False

    # ==========================================
    # FLUX 1 : TRAITEMENT PAR INTELLIGENCE ARTIFICIELLE
    # ==========================================
    if use_ai:
        if not api_key:
            st.warning(f"Veuillez entrer votre clé API {ai_provider} dans le panneau latéral pour utiliser l'IA.")
        else:
            csv_text = df_raw.head(800).to_csv(index=False) 
            if len(df_raw) > 800:
                st.warning("Fichier volumineux : seules les 800 premières lignes seront analysées par l'IA.")
                
            prompt = f"""
            Tu es un expert Cashmag. Voici les données d'un client. 
            Extraire tous les produits vendables. Déduire la RUBRIQUE la plus logique si elle manque.
            RÈGLES TVA MARTINIQUE (Clé REGROUPEMENT) :
            - Alimentaire/Boissons : "TVA_2,1"
            - Autres (Matériel, service) : "TVA_8,5"
            - Vide "" si doute.
            FORMAT: Un JSON strictement avec une liste "articles" contenant: "RUBRIQUE", "LIBELLE", "PRIX", "CODE_BARRE", "REFERENCE", "REGROUPEMENT".
            DONNÉES:
            {csv_text}
            """
            
            try:
                with st.spinner(f"{ai_provider} analyse et structure les données..."):
                    ai_result = None
                    
                    if ai_provider == "OpenAI":
                        client = OpenAI(api_key=api_key)
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            response_format={ "type": "json_object" },
                            temperature=0.1
                        )
                        ai_result = json.loads(response.choices[0].message.content)
                        
                    elif ai_provider == "Google Gemini":
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(
                            prompt,
                            generation_config=genai.GenerationConfig(
                                response_mime_type="application/json",
                                temperature=0.1,
                            )
                        )
                        ai_result = json.loads(response.text)
                    
                    if ai_result:
                        for item in ai_result.get("articles", []):
                            df_final.loc[len(df_final)] = {
                                'RUBRIQUE': item.get('RUBRIQUE', 'SANS RUBRIQUE'),
                                'ABREGE PRODUIT': item.get('LIBELLE', ''),
                                'LIBELLE PRODUIT': item.get('LIBELLE', ''),
                                'REGROUPEMENT': item.get('REGROUPEMENT', ''),
                                'CODE BARRE': item.get('CODE_BARRE', ''),
                                'CODE FOURNISSEUR': item.get('REFERENCE', ''),
                                'PRIX LIBRE (prix variable)': '',
                                'PRIX': item.get('PRIX', 0.0),
                                'PRIX 2 (A emporter)': '',
                                'PRIX 3 (Autre)': '',
                                'DESCRIPTION': ''
                            }
                        conversion_ready = True
                        st.success(f"✨ Traitement {ai_provider} terminé !")
            except Exception as e:
                st.error(f"Erreur avec l'IA ({ai_provider}) : {e}")

    # ==========================================
    # FLUX 2 : TRAITEMENT CLASSIQUE (MERCURIALE OU MAPPING)
    # ==========================================
    else:
        df_mercuriale, is_mercuriale = auto_format_mercuriale(df_raw)
        
        if is_mercuriale:
            st.success("🤖 Format 'Menu/Mercuriale' détecté (Traitement automatique sans IA) !")
            df_final['RUBRIQUE'] = df_mercuriale['RUBRIQUE']
            df_final['LIBELLE PRODUIT'] = df_mercuriale['LIBELLE PRODUIT']
            df_final['PRIX'] = df_mercuriale['PRIX']
            conversion_ready = True
            
        else:
            st.success(f"Fichier classique lu avec succès ({len(df_raw)} lignes).")
            st.subheader("⚙️ Vérifiez les correspondances")
            client_cols = ["Aucune"] + list(df_raw.columns)
            mapping = {}
            col1, col2 = st.columns(2)
            
            for i, target_col in enumerate(CASHMAG_COLS):
                guessed = guess_column(target_col, client_cols[1:])
                default_index = client_cols.index(guessed) if guessed in client_cols else 0
                with col1 if i % 2 == 0 else col2:
                    mapping[target_col] = st.selectbox(f"Cible : {target_col}", options=client_cols, index=default_index, key=target_col)

            if st.button("Lancer la conversion", type="primary"):
                for cashmag_col, client_col in mapping.items():
                    if client_col != "Aucune":
                        if cashmag_col == 'REGROUPEMENT':
                            df_final[cashmag_col] = df_raw[client_col].apply(lambda x: f"TVA_{str(x).replace('.', ',').rstrip(',0')}" if pd.notna(x) else "")
                        else:
                            df_final[cashmag_col] = df_raw[client_col]
                    else:
                        df_final[cashmag_col] = ""
                
                df_final = df_final.dropna(subset=['LIBELLE PRODUIT', 'PRIX'], how='all')
                conversion_ready = True

    # ==========================================
    # FINALISATION : REGLES CASHMAG ET TELECHARGEMENT
    # ==========================================
    if conversion_ready and not df_final.empty:
        df_final = enforce_cashmag_rules(df_final)
        
        st.write(f"**Aperçu des données prêtes pour Cashmag ({len(df_final)} articles validés) :**")
        st.dataframe(df_final.head(15))
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_final.to_excel(writer, index=False, sheet_name='Feuille1')
        output.seek(0)
        
        mode_str = ai_provider.replace(" ", "_").upper() if use_ai else "CLASSIQUE"
        st.download_button(
            label=f"⬇️ Télécharger le fichier Cashmag", 
            data=output, 
            file_name=f"IMPORT_CASHMAG_{mode_str}.xlsx"
        )