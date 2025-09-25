#!/usr/bin/env python3
"""
SEMMA 完整测试脚本
测试GPT-4o关系语义生成 + jina-embeddings-v3文本嵌入 + 完整SEMMA流程
"""

import sys
import os
from torch_geometric.data import Data
sys.path.append(os.path.dirname(__file__))

import torch
from ultra.text_semantics import (
    test_dmx_api, 
    test_jina_embeddings,
    load_or_generate_relation_semantics,
    build_relation_embeddings,
    make_inverse_by_negation
)
from ultra.tasks import build_text_relation_graph

def test_semma_pipeline():
    """
    测试完整的SEMMA管道
    """
    print("=" * 80)
    print("SEMMA 完整管道测试")
    print("=" * 80)
    
    # 测试配置
    api_key = "sk-dvJb2Dz7h6ikAnEAXP7morrMIIxTcpQN6EPKTUza7AqmSkNc"
    cache_file = "./cache/test_relation_semantics.pkl"
    
    # 模拟关系示例（简化版本用于测试）
    rel_to_example = {
        "born_in": ("person", "born_in", "place"),
        "works_for": ("person", "works_for", "company"),
        "located_in": ("city", "located_in", "country"),
        "part_of": ("district", "part_of", "city"),
        "capital_of": ("city", "capital_of", "country")
    }
    
    print(f"\n1. 测试关系语义生成 (GPT-4o)...")
    print(f"   关系数量: {len(rel_to_example)}")
    
    try:
        cleaned, descs = load_or_generate_relation_semantics(
            rel_to_example=rel_to_example,
            cache_file=cache_file,
            api_key=api_key,
            model="gpt-4o-2024-11-20",
            force_regenerate=False
        )
        
        print(f"   ✅ 成功生成语义!")
        print(f"   - 清洗后关系名: {list(cleaned.values())}")
        print(f"   - 关系描述数量: {len(descs)}")
        
        # 显示一些示例
        for i, (rel, (desc, inv_desc)) in enumerate(list(descs.items())[:3]):
            print(f"   - {rel}: '{desc}' / '{inv_desc}'")
        
    except Exception as e:
        print(f"   ❌ 关系语义生成失败: {e}")
        return False
    
    print(f"\n2. 测试文本嵌入生成 (jina-embeddings-v3)...")
    
    try:
        rel_names = list(rel_to_example.keys())
        rel_text_init = build_relation_embeddings(
            rel_names=rel_names,
            cleaned=cleaned,
            descs=descs,
            combine="COMBINED_SUM"
        )
        
        print(f"   ✅ 成功生成文本嵌入!")
        print(f"   - 嵌入形状: {rel_text_init.shape}")
        print(f"   - 嵌入类型: {rel_text_init.dtype}")
        
        # 测试逆关系嵌入
        rel_text_full = make_inverse_by_negation(rel_text_init)
        print(f"   - 逆关系嵌入形状: {rel_text_full.shape}")
        
    except Exception as e:
        print(f"   ❌ 文本嵌入生成失败: {e}")
        return False
    
    print(f"\n3. 测试文本关系图构建...")
    
    try:
        # 创建模拟图数据
        graph = Data()
        graph.num_relations = len(rel_names)
        
        # 构建文本关系图
        graph = build_text_relation_graph(
            graph, 
            rel_text_init, 
            threshold=0.8, 
            top_percent=None
        )
        
        print(f"   ✅ 成功构建文本关系图!")
        print(f"   - 节点数: {graph.text_relation_graph.num_nodes}")
        print(f"   - 边数: {graph.text_relation_graph.edge_index.shape[1]}")
        if graph.text_relation_graph.edge_weight.numel():
            print(f"   - 边权重范围: {graph.text_relation_graph.edge_weight.min(dim=0):.3f} - {graph.text_relation_graph.edge_weight.max(dim=0):.3f}")
        else:
            print("文本关系图无边!")
    except Exception as e:
        print(f"   ❌ 文本关系图构建失败: {e}")
        return False
    
    return True


def main():
    print("SEMMA 完整测试开始...")
    print("=" * 80)
    
    # # 1. 测试基础组件
    # print("\n🔧 测试基础组件...")
    
    # # 测试DMX API
    # print("\n1. 测试 DMX API 连接...")
    # api_success = test_dmx_api()
    
    # # 测试jina embeddings
    # print("\n2. 测试 jina-embeddings-v3...")
    # jina_success = test_jina_embeddings()
    
    # if not (api_success and jina_success):
    #     print("\n❌ 基础组件测试失败，请检查:")
    #     if not api_success:
    #         print("   - DMX API 连接问题")
    #     if not jina_success:
    #         print("   - jina-embeddings-v3 下载或运行问题")
    #     return
    
    # print("\n✅ 基础组件测试通过!")
    
    # 2. 测试完整管道
    print("\n🚀 测试完整SEMMA管道...")
    pipeline_success = test_semma_pipeline()
    
    # 3. 总结
    print("\n" + "=" * 80)
    print("测试结果总结")
    print("=" * 80)
    
    if pipeline_success:
        print("✅ 所有测试通过!")
        print("\n🎉 SEMMA 模块已就绪，可以运行:")
        print("   python script/run.py -c config/transductive/semma_example.yaml \\")
        print("     --dataset FB15k237 --epochs 0 --gpus [0] \\")
        print("     --ckpt ./ckpts/ultra_4g.pth")
        
        print("\n📝 注意事项:")
        print("   1. 首次运行会调用GPT-4o API生成关系语义")
        print("   2. 语义会缓存到 ./cache/ 目录，避免重复生成")
        print("   3. jina-embeddings-v3 会从HuggingFace自动下载")
        print("   4. 确保有足够的磁盘空间 (~1.5GB for jina model)")
        
    else:
        print("❌ 部分测试失败，请检查错误信息")
        print("\n🔧 故障排除:")
        print("   1. 检查网络连接")
        print("   2. 确认API密钥有效")
        print("   3. 安装依赖: pip install transformers torch torch-geometric")


if __name__ == "__main__":
    main()
