# 使いかた

起動方法

```
sh docker/build-docker.sh
sh docker/run-docker.sh
```

## train

```
python tools/umap/train.py -i /dataset/bb_error/real_test/data -o outputs/train/fruits_real
```

## test

```
python tools/umap/test.py outputs/train/fruits_real_vgg16_224 -i "/dataset/patent/eol/exp_1/data/**/**" -o outputs/test/fruits_vgg16_224/exp_1.csv
```
python tools/umap/test.py outputs/train/ -i "dataset/test" -o outputs/test/real.csv -s 224
## plot

プログラム未整理．

特許用：tools/umap/plot_all.py

# UMAPのポイントにカーソルを合わせると対応する3dmodelの画像を表示
python tools/umap/3d_model/interactive_umap_viewer.py 

---

## 3D Model UMAP Extension

このプロジェクトは、3Dモデル用のUMAPベースの埋め込み可視化システムを提供します。

### 機能概要

- **複数の特徴抽出手法**: Point Cloud サンプリング、PointNet
- **拡張可能なアーキテクチャ**: 新しい特徴抽出手法を簡単に追加可能
- **インタラクティブ可視化**: UMAPプロット上のポイントにホバーすると対応する3Dモデルの画像をリアルタイム表示
- **Blenderレンダリング**: 高品質な3Dモデル画像を動的生成
- **自動設定管理**: 訓練時の設定を自動保存・読み込み

### 特徴抽出手法

#### 1. Point Cloud サンプリング
```bash
python tools/umap/3d_model/train_3d.py --feature-extractor pointcloud -n 2048
```
- 3Dメッシュ表面から均等に点をサンプリング
- 軽量で高速
- 特徴次元: `n_points × 3`

#### 2. PointNet
```bash
python tools/umap/3d_model/train_3d.py --feature-extractor pointnet -n 2048 --pointnet-feature-dim 1024
```
- 深層学習ベースの特徴抽出
- 点群の順序に依存しない表現学習
- 特徴次元: `pointnet_feature_dim`（デフォルト: 1024）

### ファイル構成

- `train_3d.py`: 3Dモデル用UMAPモデルの訓練（複数特徴抽出手法対応）
- `test_3d.py`: 訓練済みモデルを使用して3Dモデルの埋め込みを生成（自動設定読み込み）
- `interactive_umap_viewer.py`: インタラクティブなUMAP可視化ツール（Blenderレンダリング対応）
- `render_obj.py`: Blender用3Dモデルレンダリングスクリプト
- `show_model.py`: 単一3Dモデルの表示ツール
- `plot_3d.py`: 3D埋め込みの静的プロット生成

### 使用方法

#### 1. 3D UMAP モデルの訓練

**Point Cloud サンプリング（基本）:**
```bash
python tools/umap/3d_model/train_3d.py -i data/3d_models -o outputs/train_3d/model \
  --feature-extractor pointcloud -n 2048
```

**PointNet（高度）:**
```bash
python tools/umap/3d_model/train_3d.py -i data/3d_models -o outputs/train_3d/model \
  --feature-extractor pointnet -n 2048 --pointnet-feature-dim 1024 --encoder-type mlp
```

**オプション:**
- `--feature-extractor`: 特徴抽出手法（`pointcloud`, `pointnet`）
- `--encoder-type`: エンコーダタイプ（`mlp`, `pointnet`）
- `-n, --n-points`: 各3Dモデルからサンプリングする点数
- `--pointnet-feature-dim`: PointNet特徴次元数
- `-i, --input-dir`: 3Dモデル（OBJファイル）のディレクトリ
- `-o, --output-dir`: 訓練済みモデルの保存先
- `-p, --prefix`: 3Dモデルファイルの拡張子

#### 2. 3Dモデルの埋め込み生成

**単一フォルダ:**
```bash
python tools/umap/3d_model/test_3d.py outputs/train_3d/model/PointNet_2048_1024_mlp \
  -i data/test_3d -o outputs/test_3d/embeddings_with_paths.csv
```

**複数フォルダ（フォルダごとに色分け）:**
```bash
python tools/umap/3d_model/test_3d.py outputs/train_3d/model/PointNet_2048_1024_mlp \
  -i dataset/3dmodel/fruits dataset/3dmodel/tools dataset/3dmodel/toys \
  --folder-labels "Fruits" "Tools" "Toys" \
  -o outputs/test_3d/multi_folder_embeddings.csv
```

**自動設定読み込み機能:**
- 訓練時の設定（特徴抽出手法、パラメータ）を`config.json`から自動読み込み
- 手動指定も可能（`--feature-extractor`, `--n-points`, `--pointnet-feature-dim`）

**オプション:**
- `model_dir`: 訓練済みUMAPモデルのディレクトリ
- `-i, --input-dirs`: テスト用3Dモデルのディレクトリ（複数指定可能）
- `--folder-labels`: 各フォルダのカスタムラベル（オプション、未指定時はフォルダ名を使用）
- `-o, --output-path`: 出力CSVファイルのパス
- `--max`: 処理する最大ファイル数（フォルダごと）
- `-p, --parallel`: 並列処理数

#### 3. インタラクティブ可視化

```bash
python tools/umap/3d_model/interactive_umap_viewer.py
```

**機能:**
- フォルダラベルによる自動色分け表示
- UMAPプロット上のポイントにホバーすると対応する3DモデルがBlenderで自動レンダリング
- フォルダ情報とモデル名の表示
- レンダリング状況の表示

#### 4. 静的プロット生成

```bash
python tools/umap/3d_model/plot_3d.py -i outputs/test_3d/multi_folder_embeddings.csv \
  -o outputs/plot_3d --title "Multi-Folder 3D Model Analysis"
```

**機能:**
- フォルダラベルによる色分けプロット
- カスタマイズ可能なタイトルとサイズ
- 詳細な統計情報付きサマリー生成

#### 5. 単一モデル表示

```bash
python tools/umap/3d_model/show_model.py path/to/model.obj
```

### 拡張性について

#### 新しい特徴抽出手法の追加

1. `FeatureExtractor`を継承したクラスを作成:
```python
class CustomExtractor(FeatureExtractor):
    def extract_features(self, mesh: trimesh.Trimesh, **kwargs) -> np.ndarray:
        # カスタム特徴抽出ロジック
        pass
    
    def get_feature_dim(self, **kwargs) -> int:
        # 特徴次元数を返す
        pass
    
    def get_name(self) -> str:
        # 識別名を返す
        pass
```

2. `create_feature_extractor`関数に追加
3. `parse_args`の選択肢に追加

### アーキテクチャの特徴

- **抽象基底クラス**: `FeatureExtractor`で統一インターフェース
- **ファクトリパターン**: `EncoderFactory`で柔軟なエンコーダ生成
- **設定管理**: JSON形式で訓練・テスト設定を自動保存
- **エラーハンドリング**: 堅牢なメッシュ処理とエラー回復
- **型ヒント**: 完全な型注針でコードの可読性向上

### 必要な依存関係

- **Python パッケージ**: trimesh, dash, pandas, plotly, PIL, tensorflow
- **外部ツール**: Blender（コマンドライン実行可能な状態）
- **システム**: OpenGL対応環境

### パフォーマンス比較

| 特徴抽出手法 | 速度 | メモリ使用量 | 表現力 | 用途 |
|------------|------|------------|-------|------|
| Point Cloud | 高速 | 低 | 中 | 高速プロトタイピング |
| PointNet | 中速 | 高 | 高 | 高精度分析 |

### Docker環境での設定

Dockerfileには以下の3Dレンダリング用ライブラリが含まれています:

```dockerfile
RUN apt-get update && apt-get install -y \
    libglu1-mesa \
    libosmesa6 \
    libopengl0 \
    libglx0 \
    libegl1 \
    xvfb
```

### トラブルシューティング

**特徴抽出関連:**
- PointNet使用時のメモリ不足: `--pointnet-feature-dim`を小さくする
- 訓練時間が長い: Point Cloudサンプリングを試す
- 設定ミスマッチ: `config.json`の内容を確認

**その他:**
- Blenderが見つからない場合: `blender --version`で確認
- レンダリングエラー: コンソール出力でBlenderのエラーメッセージを確認

### 出力形式

- **埋め込みCSV**: `model_path`, `umap_0`, `umap_1`カラムを含む
- **設定ファイル**: `config.json`（訓練設定）、`*_config.json`（テスト設定）
- **レンダリング画像**: PNG形式、800x600解像度
- **プロット**: 高解像度PNG（300 DPI）