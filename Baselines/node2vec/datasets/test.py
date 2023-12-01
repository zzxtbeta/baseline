import networkx as nx
from node2vec import Node2Vec

# 创建一个图
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)
# 读取带权重的边列表
graph = nx.read_weighted_edgelist('bitcoinotc_sorted-1.csv', delimiter='\t', nodetype=int)

# 预计算概率并生成随机游走路径 - **仅在 Windows 上，如果使用 workers=1 才能正常运行**
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)  # 对于大图，可以使用 temp_folder 参数

# 嵌入节点
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # 可传递任何被 gensim.Word2Vec 接受的关键字参数，`dimensions` 和 `workers` 会自动从 Node2Vec 构造函数传递过来

# 查找最相似的节点
model.wv.most_similar('2')  # 输出的节点名称总是字符串

# 保存嵌入以备后用
EMBEDDING_FILENAME = "../emb/otc_embedding_filename.txt"
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# 保存模型以备后用
EMBEDDING_MODEL_FILENAME = "../emb/otc_embedding_model_filename.model"
model.save(EMBEDDING_MODEL_FILENAME)

# 使用Hadamard方法嵌入边
from node2vec.edges import HadamardEmbedder

edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# 实时查找嵌入 - 在这里我们传递正常的元组
edges_embs[('1', '2')]
''' 输出
array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
       ... ... ....
       ..................................................................],
      dtype=float32)
'''

# 获取所有边的独立 KeyedVectors 实例 - 对于大型网络，请小心使用可能会很大
edges_kv = edges_embs.as_keyed_vectors()

# 查找最相似的边 - 这次元组必须被排序并转换为字符串
edges_kv.most_similar(str(('1', '2')))

# 保存嵌入以备后用
EDGES_EMBEDDING_FILENAME = "../emb/otc_edges_embedding_filename.txt"
edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)

