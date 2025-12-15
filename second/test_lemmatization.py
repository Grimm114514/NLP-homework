"""
演示词形还原功能
安装依赖: pip install nltk
"""

from data import lemmatize_tokens, USE_LEMMATIZATION

if __name__ == "__main__":
    if not USE_LEMMATIZATION:
        print("请先安装 NLTK: pip install nltk")
        exit()
    
    # 测试词形还原
    test_words = [
        'running', 'runs', 'ran', 'run',
        'going', 'goes', 'went', 'gone', 'go',
        'better', 'best', 'good',
        'children', 'child',
        'cacti', 'cactus',
        'thinking', 'thinks', 'thought', 'think',
        'wrote', 'writing', 'writes', 'written', 'write'
    ]
    
    print("=" * 60)
    print("词形还原测试")
    print("=" * 60)
    
    lemmatized = lemmatize_tokens(test_words)
    
    print(f"\n{'原词':<15} -> {'还原后':<15}")
    print("-" * 35)
    for orig, lemma in zip(test_words, lemmatized):
        marker = "✓" if orig != lemma else " "
        print(f"{orig:<15} -> {lemma:<15} {marker}")
    
    print("\n统计:")
    print(f"原始唯一词数: {len(set(test_words))}")
    print(f"还原后唯一词数: {len(set(lemmatized))}")
