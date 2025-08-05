#!/usr/bin/env python3
"""
VHACDファイル削除ツール

指定されたディレクトリとその階層下にある"vhacd"を含むファイル名のファイルを削除します。
"""

import os
import argparse
import glob
from pathlib import Path


def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="指定ディレクトリから'vhacd'を含むファイルを再帰的に削除します"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="検索対象のディレクトリパス"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際に削除せずに、削除対象ファイルのリストのみ表示"
    )
    parser.add_argument(
        "--pattern",
        default="*vhacd*",
        help="検索パターン（デフォルト: *vhacd*）"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="削除前に確認プロンプトを表示"
    )
    return parser.parse_args()


def find_vhacd_files(directory, pattern="*vhacd*"):
    """
    指定ディレクトリからvhacdを含むファイルを再帰的に検索
    
    Args:
        directory (str): 検索対象ディレクトリ
        pattern (str): 検索パターン
        
    Returns:
        list: 見つかったファイルパスのリスト
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"エラー: ディレクトリ '{directory}' が存在しません。")
        return []
    
    if not directory_path.is_dir():
        print(f"エラー: '{directory}' はディレクトリではありません。")
        return []
    
    # 再帰的にファイルを検索
    vhacd_files = []
    for file_path in directory_path.rglob(pattern):
        if file_path.is_file():
            vhacd_files.append(file_path)
    
    return sorted(vhacd_files)


def format_file_size(size_bytes):
    """ファイルサイズを人間が読みやすい形式にフォーマット"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def delete_files(file_list, dry_run=False, confirm=False):
    """
    ファイルリストのファイルを削除
    
    Args:
        file_list (list): 削除対象ファイルのリスト
        dry_run (bool): ドライランモード（実際に削除しない）
        confirm (bool): 削除前に確認
        
    Returns:
        tuple: (削除されたファイル数, 削除されたファイルの総サイズ)
    """
    if not file_list:
        print("削除対象のファイルが見つかりませんでした。")
        return 0, 0
    
    total_size = 0
    deleted_count = 0
    
    # ファイル情報の表示
    print(f"\n見つかったファイル数: {len(file_list)}")
    print("=" * 80)
    
    for file_path in file_list:
        try:
            file_size = file_path.stat().st_size
            total_size += file_size
            size_str = format_file_size(file_size)
            print(f"{size_str:>10} | {file_path}")
        except (OSError, FileNotFoundError) as e:
            print(f"エラー: {file_path} - {e}")
    
    print("=" * 80)
    print(f"総サイズ: {format_file_size(total_size)}")
    
    if dry_run:
        print("\n[ドライランモード] 実際の削除は行われません。")
        return len(file_list), total_size
    
    if confirm:
        response = input(f"\n{len(file_list)}個のファイルを削除しますか？ (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("削除がキャンセルされました。")
            return 0, 0
    
    # 実際の削除処理
    print("\nファイルを削除中...")
    for file_path in file_list:
        try:
            file_path.unlink()
            deleted_count += 1
            print(f"削除: {file_path}")
        except (OSError, FileNotFoundError) as e:
            print(f"削除エラー: {file_path} - {e}")
    
    return deleted_count, total_size


def main():
    """メイン処理"""
    args = parse_args()
    
    print(f"検索ディレクトリ: {args.directory}")
    print(f"検索パターン: {args.pattern}")
    
    if args.dry_run:
        print("[ドライランモード] 実際の削除は行われません。")
    
    # vhacdファイルの検索
    print("\nvhacdファイルを検索中...")
    vhacd_files = find_vhacd_files(args.directory, args.pattern)
    
    # ファイルの削除
    deleted_count, total_size = delete_files(
        vhacd_files, 
        dry_run=args.dry_run, 
        confirm=args.confirm
    )
    
    # 結果の表示
    if args.dry_run:
        print(f"\n[結果] {len(vhacd_files)}個のファイルが削除対象です（総サイズ: {format_file_size(total_size)}）")
    else:
        print(f"\n[結果] {deleted_count}個のファイルを削除しました（総サイズ: {format_file_size(total_size)}）")


if __name__ == "__main__":
    main() 