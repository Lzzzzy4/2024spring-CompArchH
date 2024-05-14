# Lab3实验报告

## PB21000164 来泽远

## Cache更改

```verilog
reg [WAY_CNT - 1:0]cache_hit_hot = 0; // one-hot
always @ (*) begin 
    for (integer i = 0; i < WAY_CNT; i = i + 1) begin
        if(valid[set_addr][i] && cache_tags[set_addr][i] == tag_addr) 
            cache_hit_hot[i] = 1'b1;
        else
            cache_hit_hot[i] = 1'b0;
    end
end

wire cache_hit;
assign cache_hit = |cache_hit_hot;

reg [WAY_ADDR_LEN - 1:0] hit_addr = 0;
always @ (*) begin
    integer i;
    for (i = 0; i < WAY_CNT; i = i + 1) begin
        if (cache_hit_hot[i] == 1'b1) begin
            hit_addr = i;
        end
    end
end

reg [WAY_ADDR_LEN - 1:0] replace_addr = 0;
reg [WAY_ADDR_LEN - 1:0] mem_rd_replace_addr = 0;

reg [WAY_ADDR_LEN - 1:0] qe_cnt    [SET_SIZE][WAY_CNT];
always @ (*) begin
    for (integer i = 0; i < WAY_CNT; i = i + 1) begin
        if (qe_cnt[set_addr][i] == 0) begin
            replace_addr = i;
        end
    end
    // replace_addr = qe_cnt[set_addr][0];
end
always @ (posedge clk or posedge rst) begin  
    if (rst) begin
        for(integer i = 0; i < SET_SIZE; i++) begin
            for(integer j = 0; j < WAY_CNT; j++) begin
                qe_cnt[i][j] = j;
            end
        end
    end
    else begin
        case(cache_stat)
        IDLE: begin
            if (cache_hit && (wr_req || rd_req)) begin
                `ifdef LRU
                for (integer i = 0; i < WAY_CNT; i = i + 1) begin
                    if (i == hit_addr) begin
                        qe_cnt[set_addr][i] <= WAY_CNT - 1;
                    end
                    else if (qe_cnt[set_addr][i] > qe_cnt[set_addr][hit_addr]) begin
                        qe_cnt[set_addr][i] <= qe_cnt[set_addr][i] - 1;
                    end
                end
                `endif
            end
        end
        SWAP_IN_OK:begin
            for (integer i = 0; i < WAY_CNT; i = i + 1) begin
                if (i == mem_rd_replace_addr) begin
                    qe_cnt[set_addr][i] <= WAY_CNT - 1;
                end
                else if(qe_cnt[set_addr][i] > qe_cnt[set_addr][hit_addr]) begin
                    qe_cnt[set_addr][i] <= qe_cnt[set_addr][i] - 1;
                end
            end
        end
        default:begin
        end
        endcase
    end
end
```

核心代码如上，对于`cache_mem、cache_tags、valid、dirty`只需要加一维即可，其余无需修改。

关于替换策略，对每一个line维护一个队列计数器，即`qe_cnt`。组相联对应line之间初始值为`range(0,CNT_WAY)`。

一次更新操作表现为，对于最近操作的一个line，将其计数器置为最大值`WAY_CNT - 1`，其余比其原始值大的计数器减1。

LRU在有访问且hit和swapin时更新，FIFO仅在swapin时更新。

## 实验结果

LINE_ADDR_LEN + SET_ADDR_LEN + TAG_ADDR_LEN = 12

quicksort: N = 256

matmul: N = 4

(clk_quicksort)/clk_(matmul)

| LINE | SET  | TAG  | WAY  | strategy | clk          | MissRate/% | LUT  | FF   |
| ---- | ---- | ---- | ---- | -------- | ------------ | ---------- | ---- | ---- |
| 3    | 3    | 6    | 3    | LRU      | 53105/258323 | 2.80/27.72 | 3324 | 7340 |
| 3    | 3    | 6    | 3    | FIFO     | 57127/305732 | 3.91/32.62 | 3018 | 7335 |
| 2    | 4    | 6    | 3    | LRU      | 73648/321062 | 6.09/33.23 | 3263 | 7197 |
| 2    | 4    | 6    | 3    | FIFO     | 75169/332254 | 7.81/35.09 | 3148 | 7196 |
| 4    | 2    | 6    | 3    | LRU      | 60712/323698 | 3.75/34.32 | 5721 | 7994 |
| 3    | 3    | 6    | 2    | LRU      | 63930/305732 | 4.44/32.63 | 1974 | 5191 |
| 3    | 3    | 6    | 4    | LRU      | 56238/305732 | 3.43/32.62 | 4631 | 9469 |

设置估价函数如下：
$$
\begin{align}
score &= \frac{性能指标}{资源指标} \\
&= (\frac{clk_{sort,max}}{clk_{sort}} + \frac{clk_{mat,max}}{clk_{mat}})/(\frac{LUT}{LUT_{max}} + \frac{FF}{FF_{max}})
\end{align}
$$
计算得

| LINE | SET  | TAG  | WAY  | strategy | score |
| ---- | ---- | ---- | ---- | -------- | ----- |
| 3    | 3    | 6    | 3    | LRU      | 1.99  |
| 3    | 3    | 6    | 3    | FIFO     | 1.85  |
| 2    | 4    | 6    | 3    | LRU      | 1.54  |
| 2    | 4    | 6    | 3    | FIFO     | 1.52  |
| 4    | 2    | 6    | 3    | LRU      | 1.22  |
| 3    | 3    | 6    | 2    | LRU      | 2.52  |
| 3    | 3    | 6    | 4    | LRU      | 1.33  |

给出较优配置为`LINE=3,SET=3,TAG=3,WAY=2,strategy=LRU`。

该配置下的仿真波形与资源用量报告截图如下（cnt为时钟周期计数）：

![image-20240514221200056](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240514221200056.png)

![image-20240514221355258](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20240514221355258.png)





其较为感性的认知是，该配置在性能牺牲很小的前提下获得了很大的资源节省。

对于其中每一个属性分析如下：

1. LINE。LINE过大会导致内存的读写浪费时间，比如某一个line中只有一个是被频繁用到的，其他的读取额外浪费了$(2^4 -1) \times50$个周期。同样的，脏块仅对line标记，一个word写脏导致整个line被重新写入。LINE过小会导致cache失去局部性，比如一个内存地址相邻的数组，并对其保证一定的访问频率。LINE过小会导致该数组被拆分为好几个line存放在不同的set中，部分因访问过少被换回，再需访问时又要访问主存。
2. SET。该属性与LINE有类似性，无非就是分别对高低位地址作用。如快速排序等局部性强的程序，可以LINE的重要性大于SET，如4/2配置较2/4在快排上的表现更好。又如矩阵乘法，不仅有局部的一维递增$(a[i][j],a[i][j+1])$，更有相隔固定间隔的二维递增$(a[i][i],a[i+1][j])$。此时SET作用在更高维地址的优势显现，2/4配置较4/2配置在矩阵访问的表现上更好。
3. WAY。WAY的增大带来的提升具有边际效应。WAY的增大并不能使cache中缓存的地址空间变大，而LINE和SET均可。WAY的增大可变相理解为为每一个line配置一个victim cache，且该cache容量变大。同样的victim cache也具有边际效应。即许久不用的值存放在victim cache和存放在主存中的效果是相同的。
4. strategy。LRU基本优于FIFO，这是符合直觉的。两者更新规则相同，而LRU更新频率高，每一次访问更新一次，其计数器更加贴近真实的数据访问模式。不过可以修改为伪LRU，伪LRU在WAY小时的效果极贴近LRU，还可以优化掉减法器（不需要使用队列），大大减少了LUT用量。
5. LUT。主要源自替换算法的复杂性。LUT主要来自于组合逻辑，如上文提及的减法器，以及cache中的选择器都要用到LUT。
6. FF。与$2^{LINE}*2^{SET}*WAY$成正比，也即cache的存储单元。可以看到表中的FF基本符合这个规律。

## 实验收获

对cache的实现有了更加深刻的理解，掌握了工程细节。同时也熟悉了如何将cache适配流水线，包括hazard单元的配合。

通过实现调参，更好的理解了cache的配置参数和替换策略，对于性能以及资源用量与cache参数之间的关系有了初步的了解。

## 改进意见

可以将生成数据和指令sv文件的脚本整合一下，变成一个文件。以及在我的环境下，`\# -*- coding: utf-8 -*-\`是失效的，无法正常编码为utf-8。建议不使用管道而使用`with open('cache_tb.sv', 'w', encoding='utf-8') as f:`。