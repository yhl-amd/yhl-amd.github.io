# Study notes

- [DeepSeek V4 KV cache · vLLM 与 ATOM](dsv4-kv-cache/)：可交互讲义。← → 翻页，O 打开目录，R 切换阅读模式（附讲稿），页面底部的“源码”按钮可查看固定版本的源码片段。
- [DP Prefill 代价均衡 · DSV4-Pro DP attention](dsv4-dp-prefill-balance/)：按注意力代价削掉每步最慢那张卡的 prefill；代价模型、调度器实现、和会话迁移（spill）的对比、实验数据。
- [DeepSeek-V4.1-Flash 图解](dsv41-flash/)：V4.1-Flash 相对 V4 改了什么。概览、重点（CED、CSA2 跨层共享、分层 indexer、FP4 KV、SWA Bounded Replay）、对照论文图的精讲，附可交互的小演示。
- [ATOM 中的 MegaMoE 实现](atom-megamoe/)：从 MoE 计算讲起，分三部分：概览、重点（4 次 launch、fixed-slot 与 compact 两种 dispatch、实例选择、DP pad 行屏蔽、GEMM 融合）、代码精讲与优化记录。基于 ATOM da0e87d / aiter 2274275。
- [LMCache MP 模式的 RPC 开销 · MiniMax-M3 TP4](lmcache-mp-overheads/)：MP 模式为什么比进程内慢。原版 server 每次 RPC 的三笔开销（销毁导入的 IPC event 要等 GPU、每个 RPC 带整段 token 前缀、ObjectKey 逐个构造）的图解和举例，三项原型改动的端到端消融（原版、三项全开、inproc 各重复 3 次）和只压 server 的测试。基于 LMCache 0.5.6.dev139 / ATOM 6e9b3e1c5。
