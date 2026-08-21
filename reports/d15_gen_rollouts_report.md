# d15_gen_rollouts_report — 37,079 条 longform rollout 生成收官报告

- 资产:/scratch/li96/mz9869/dflash_data/longform_fixed_v2/rollouts_px501760/
  (8 × JSONL + manifest.json,**complete=true, n_cached=37,079/37,079**)
- 生成配置:Qwen2.5-VL-7B-Instruct greedy(rep_penalty=1.0 显式压回、
  单 EOS 151645、幻影词表屏蔽)、max_pixels=501760(longform
  PREPROCESS_SPEC)、max_new=512、bs=8、dgxa100;code_commit d892b9e
- 审查链:必审包(REVIEW_NOTES)→ 三处修订(c72fbdc)→ 冒烟
  176710098(E=5.0 过门)→ 用户放行 → 8 独立作业(submit 脚本,弃 array)

## 1. 逐 shard 台账(哨兵 + PBS 日志对账)

| shard | job id | rows | n_valid | walltime | SU | fail | 续跑 | dirty |
|---|---|---|---|---|---|---|---|---|
| s0 | 176722377 | [0, 4635) | 4,635/4,635 | 02:41:07 | 193.34 | 0 | 0 | false |
| s1 | 176722378 | [4635, 9270) | 4,635/4,635 | 02:41:43 | 194.06 | 0 | 0 | false |
| s2 | 176722379 | [9270, 13905) | 4,635/4,635 | 02:45:45 | 198.90 | 0 | 0 | false |
| s3 | 176722380 | [13905, 18540) | 4,635/4,635 | 02:45:47 | 198.94 | 0 | 0 | false |
| s4 | 176722382 | [18540, 23175) | 4,635/4,635 | 02:44:08 | 196.96 | 0 | 0 | false |
| s5 | 176722383 | [23175, 27810) | 4,635/4,635 | 02:43:46 | 196.52 | 0 | 0 | false |
| s6 | 176722384 | [27810, 32445) | 4,635/4,635 | 02:50:47 | 204.94 | 0 | 0 | false |
| s7 | 176722385 | [32445, 37079) | 4,634/4,634 | 02:51:09 | 205.38 | 0 | 0 | false |

8/8 exit=0、零失败、零续跑(全部单次跑通,断点续跑机制未被触发——
其正确性由冒烟 drill 三验背书,见 §3)。生成段吞吐 160.6–169.9 min/shard,
批间一致性好(±3%)。

## 2. 长度分布(全池 37,079,按源分层)

| 源 | n | mean | med | p10 | p90 | max | eos_hit | trunc@512 |
|---|---|---|---|---|---|---|---|---|
| DOCCI | 9,647 | 365.7 | 332 | 226 | 512 | 512 | 74.6% | 25.4% |
| DetailCaps | 4,868 | 381.7 | 422 | 201 | 512 | 512 | 73.0% | 27.0% |
| SP | 14,564 | 404.0 | 448 | 230 | 512 | 512 | 71.9% | 28.1% |
| LN-OI | 8,000 | 383.5 | 412 | 216 | 512 | 512 | 68.6% | 31.4% |
| **全池** | **37,079** | **386.7** | **422** | 222 | 512 | 512 | **72.0%** | **28.0%** |

(trunc@512 定义 = eos_hit=false;n_tokens=512 且自然收 EOS 的边界行
不计入,故 n=512 行数 10,463 略高于截断数。)

对照 pilot 预期:v1.1(px 同 spec,n=50 分层)L̄=371.9、trunc@512=26%
→ 全量 L̄=386.7(+4.0%)、28.0%(+2pp),分布形态与 pilot 一致,
无源级异常。约 28% 的 rollout 在 512 测量上限处截断,真实长度右尾未知
——预注册条款已裁定如实登记,入档为已知测量局限(pilot 报告 §1 同款)。

- **总 rollout token:Σ L = 14,337,784**
- 训练 token 量对照旧线:旧线 35k 缓存 rollout 均长 ~103 → 新线 386.7,
  单条训练信号 ~3.75×

## 3. 谱系登记

### 3a. 冒烟 drill 64/64 全同的因果说明(非普适性质,防误引)

冒烟断点演练中,被杀重续臂与不间断臂 64/64 token 逐行全同。**因果**:
kill 落点恰好写完 40 行,而 40 为 batch_size=8 的整倍数 → 续跑从行 40
起重组的批与不间断跑的批**逐一同构** → fp16 数值路径完全一致 → 确定
性复现。**反面提醒(登记)**:全量若发生真实中断,断点一般不落在批边界
上,续跑批组成会变——这属于已登记的批内 fp16 扰动类别(改具体 token
路径,不改宏观行为,见 3b),**不得**将"续跑必逐 token 全同"当作
可复现性承诺引用。本次全量 8 单零续跑,该情形未实际发生。

### 3b. bs=8 vs bs=1 对照标定(非门)

同 64 行配对:逐行 token 全同 28/64(43.8%);宏观形态不变——均长
385.7 vs 376.4(+2.5%),eos_hit 两臂完全同为 41/64。结论:批内 fp16
扰动改变措辞级 token 路径,不改变长度/终止分布,与 drift attribution
的"前向形态差异非 bug"结论同类。本资产以 **bs=8 产物为锁定正典**
(artifact 锁定制:训练目标即该批量形态下的实际序列,谱系由逐行
sha256 + manifest 固定,不依赖与 bs=1 的逐 token 一致性)。

## 4. SU 对账

| 项 | SU |
|---|---|
| 8 shard 全量(PBS 计费实数) | **1,589.04** |
| 预算区间(预注册) | [1,290, 2,581] ✓ |
| 冒烟预测(E=5.0 外推 1,548) | 偏差 +2.6% |
| 附:冒烟 176710098 | 22.2 |
| 附:pilot(v1 + v1.1 px 重跑) | ~30 |

外推链条(R₁ pilot → E 配对冒烟 → 全量)三级传递误差 <3%,预算方法
论有效性入档。

## 5. 下游交接(队列②)

- 抽取端三级先验就绪:manifest complete=true / 逐文件 sha256 在
  manifest / 逐行 sha256 在行内;extract_ctx_longform.load_locked_rollouts
  为第一读者
- 体积回填(budget §2 条款):T̄ = P̄(450.5,pilot v1.1)+ L̄(386.7,
  本报告实测)= 837.2 → 缓存外推 **1,113 GB ≈ 1.04 TiB**,对预算
  1,093 GB 偏 +1.8%,**±5% 内,预算不改**;算力外推 31.0M tok ÷ 1,921
  ≈ 4.49 GPU·h ≈ **323 SU**(预算预期 330 内)
- manifest 元数据冷备:archives/longform_rollouts_px501760_manifest_*.tar.gz
  (manifest + 8 哨兵;JSONL 本体不入 tar,由 scratch 正本 + 行级哈希保护)

## 附录:抽样人工过目(每源 2 条,行选取=源块首行+中位行,确定性)

### docci · idx 0 · docci/train_04996 · L=466 · eos_hit=True

> The image depicts a large, colorful sign for an event called the "Blue Genie Art Bazaar." The sign is mounted on the side of a building, which appears to be a reddish-brown brick structure. The sign itself is predominantly blue with a gradient effect, and it features a large, stylized genie figure in the upper portion.
> 
> The genie figure is a prominent, animated character with a blue skin tone, a beard, and a traditional genie outfit that includes a golden robe and a golden hat. The genie is holding a golden lamp in one hand, which has the letters "bga" inscribed on it. The genie's other hand is raised in a gesture that could be interpreted as a wave or a gesture of welcome. The genie's pose is dynamic and expressive, adding a lively and inviting feel to the sign.
> 
> Below the genie figure, the text "BLUE GENIE ART BAZAAR" is prominently displayed in bold, yellow and black letters. The word "BLUE" is in yellow, while "GENIE," "ART," and "BAZAAR" are in black. The text is designed to be eye-catching and easy to read from a distance.
> 
> To the left of the text, there is a graphic of a golden genie lamp, which is a classic symbol associated with genies and wishes. The lamp is depicted in a simple, cartoonish style, complementing the overall playful and whimsical theme of the sign.
> 
> The background of the image shows a cloudy sky, indicating that the photo was taken outdoors on a day with overcast weather. The lighting is natural, and the colors are vibrant, making the sign stand out against the more muted tones of the sky and the building.
> 
> The relationship between the genie figure and the text is clear: the genie is the central character of the event, and the text identifies the event as the "Blue Genie Art Bazaar." The genie's pose and the lamp it holds suggest a connection to the magical and creative themes of the event, which is likely an art fair or exhibition.
> 
> Overall, the image conveys a sense of fun and creativity, with the genie figure and the lamp serving as symbols of magic and artistry. The sign is designed to attract attention and convey the nature of the event in a visually appealing way.

### docci · idx 4823 · docci/train_01009 · L=374 · eos_hit=True

> The image shows a cozy indoor scene featuring a window sill with three stuffed animals placed on it. The window is a standard double-hung type with a white frame and a clear glass pane. The window is closed, and the reflection in the glass shows a blurred image of what appears to be a person and possibly some furniture, suggesting the photo was taken from inside a room looking out.
> 
> On the window sill, there are three stuffed animals:
> 
> 1. **Leftmost Stuffed Animal**: This is a light beige or cream-colored stuffed animal, possibly resembling a dog or a similar animal. It has a soft, plush texture and appears to be lying down with its head resting on the sill.
> 
> 2. **Middle Stuffed Animal**: This is a white stuffed animal, likely a polar bear, given its black nose and black feet. It is positioned in the center and is lying down as well, with its head resting on the sill.
> 
> 3. **Rightmost Stuffed Animal**: This is a brown stuffed animal, possibly resembling a bear or a similar animal. It is positioned on the far right and is also lying down, with its head resting on the sill.
> 
> The window sill is white and contrasts with the beige wall surrounding it. The wall has a smooth texture and a warm tone. To the left of the window, there is a portion of a countertop visible, which appears to be made of a dark, speckled material, possibly granite or a similar stone. A white electrical outlet is also visible on the wall to the left of the countertop.
> 
> The overall scene gives a sense of comfort and warmth, with the stuffed animals adding a playful and homely touch to the setting. The lighting in the room is soft, and the image appears to have been taken in the evening or at night, as suggested by the darkness visible through the window.

### detailcaps · idx 9647 · detailcaps/5643214 · L=490 · eos_hit=True

> This image depicts a natural landscape featuring a rocky terrain and a tree. Here's a detailed description:
> 
> 1. **Main Objects:**
>    - **Rocky Terrain:** The foreground is dominated by a large, rugged rock formation. The rocks are predominantly gray with patches of white, suggesting possible mineral deposits or weathering. The surface of the rocks is uneven, with visible cracks and crevices.
>    - **Tree:** A single tree stands prominently in the middle of the image. It has a relatively thin trunk and a sparse canopy of green leaves, indicating it might be a young or stunted tree. The tree appears to be growing on or near the rocky terrain.
> 
> 2. **Scene Context:**
>    - The setting appears to be a mountainous or hilly area, as suggested by the presence of the rocky terrain and the tree's position on a slope. The background is filled with more greenery, indicating a forested or densely vegetated area.
>    - The sky is overcast, with a grayish hue, suggesting cloudy weather. There is no direct sunlight visible, which contributes to the muted lighting in the image.
> 
> 3. **Relationships:**
>    - The tree is positioned centrally in the image, slightly elevated above the rocky terrain. It seems to be growing out of or near the rocks, possibly indicating a struggle for survival in a harsh environment.
>    - The rocks and the tree are in close proximity, with the tree appearing to be supported or partially rooted in the rocky surface.
> 
> 4. **Visual Attributes:**
>    - The colors in the image are predominantly natural and earthy. The rocks are gray and white, the tree has green leaves, and the background is a mix of green from the vegetation and the gray of the sky.
>    - The texture of the rocks is rough and detailed, with visible cracks and crevices, adding to the ruggedness of the scene.
>    - The tree's leaves are sparse, and the branches are thin, giving it a delicate appearance.
> 
> 5. **Uncertainty:**
>    - The exact location of the image is uncertain, as there are no visible signs or landmarks to identify the specific place.
>    - The age of the tree and the exact nature of the rocks are also uncertain, as there are no additional context clues to provide more information.
> 
> Overall, the image captures a serene yet rugged natural scene, highlighting the resilience of life in challenging environments.

### detailcaps · idx 12081 · detailcaps/2046688669_a801eb664b · L=267 · eos_hit=True

> The image depicts a fantastical scene featuring a white, mythical creature that appears to be a lion with wings, reminiscent of a griffin. The creature is captured in mid-flight, with its wings spread wide and its body angled downward, suggesting it is descending towards the ground. The lion's mane is flowing, adding to the sense of motion and grace.
> 
> The background is a natural setting, likely a forest or a wooded area, with dense foliage and trees. The ground is covered with grass and scattered with rocks and pebbles, indicating a natural, untamed environment. The lighting in the image is soft and diffused, giving the scene a dreamy, almost ethereal quality. The colors are muted, with shades of green, brown, and gray dominating the palette, which enhances the mystical atmosphere.
> 
> The creature's wings are large and appear to be made of feathers, which is a common feature in depictions of griffins or other mythical winged lions. The overall composition of the image suggests a narrative of a mythical creature in its natural habitat, possibly on a journey or a hunt. The scene does not contain any other objects or creatures, which places the focus entirely on the flying lion. The image evokes a sense of wonder and imagination, inviting the viewer to ponder the story behind this fantastical creature.

### sp · idx 14515 · sp/vg_2411395 · L=175 · eos_hit=True

> The image shows a bright yellow fire hydrant with red caps on the top and middle sections. The hydrant is positioned against a white wall with visible signs of wear, such as cracks and discoloration. The wall appears to be part of a building, and there is a small, rectangular, white box mounted on it, possibly a utility box or a light fixture. The ground in front of the hydrant is a dark, paved surface, and there is a small, yellow object near the base of the hydrant, which could be a cap or a part of the hydrant itself. The overall scene suggests an urban or suburban setting, with the hydrant serving as a functional and colorful element in the environment. The lighting in the image is natural, and the angle of the photograph gives a slightly distorted perspective, emphasizing the hydrant as the focal point.

### sp · idx 21797 · sp/vg_2377645 · L=456 · eos_hit=True

> The image depicts a multi-tiered wedding cake, beautifully decorated with fresh fruits and flowers. Here's a detailed description:
> 
> 1. **Cake Structure**:
>    - The cake is tiered, consisting of four distinct layers.
>    - Each layer is separated by a layer of cream or frosting, which is visible between the tiers.
> 
> 2. **Decorations**:
>    - The cake is adorned with an assortment of fresh fruits, including blackberries, raspberries, strawberries, and red currants.
>    - There are also sprigs of green leaves, likely mint or basil, interspersed among the fruits.
>    - Pink and purple roses are placed at the top and around the sides of the cake, adding a touch of elegance and color contrast.
>    - The overall presentation is natural and rustic, with the fruits and flowers giving the cake a fresh and organic look.
> 
> 3. **Background and Context**:
>    - The cake is placed on a white tablecloth, which is partially visible in the foreground.
>    - In the background, there is a blurred figure of a person, possibly a bride, wearing a white dress with a lace collar. This suggests that the setting is a wedding or a similar celebratory event.
>    - The background also includes a dark, wicker chair, indicating an outdoor or semi-outdoor venue.
> 
> 4. **Additional Elements**:
>    - A knife is placed on the table to the right of the cake, likely for cutting the cake.
>    - A small bouquet of flowers, including roses and other blooms, is positioned in the foreground, adding to the festive atmosphere.
> 
> 5. **Lighting and Focus**:
>    - The lighting is natural, possibly from daylight, as it highlights the vibrant colors of the fruits and flowers.
>    - The focus is primarily on the cake, with the background slightly blurred, drawing attention to the cake as the main subject.
> 
> 6. **Uncertainty**:
>    - The exact location and specific details of the event are not clear from the image alone.
>    - The person in the background is not clearly identifiable, and their role in the event is not specified.
> 
> Overall, the image captures a beautifully decorated wedding cake, likely at a wedding reception, with a focus on natural and fresh elements.

### ln · idx 29079 · ln/oi_6513afa881ed9557 · L=512 · eos_hit=False

> This image depicts the iconic Pike Place Market in Seattle, Washington. The scene is bustling with activity, showcasing the vibrant atmosphere of the market.
> 
> ### Main Objects:
> 1. **Neon Signs**:
>    - The most prominent feature is the large, illuminated neon sign that reads "PUBLIC MARKET" in red letters. Below it, another neon sign reads "FARMERS MARKET" in yellow letters, with a green outline.
>    - The neon signs are mounted on a metal frame, which is part of the market's entrance structure.
> 
> 2. **Clock**:
>    - Above the neon signs, there is a large, round clock with a red and white face. The clock is mounted on a metal structure and is a notable landmark of the market.
> 
> 3. **Market Entrance**:
>    - The entrance to the market is a covered area with glass windows and doors. Inside, various stalls and shops are visible, displaying a variety of goods such as produce, clothing, and other items.
> 
> 4. **People**:
>    - The market is crowded with people. Some are walking, while others are standing and browsing the stalls. The crowd appears diverse, with individuals of various ages and styles of dress.
> 
> 5. **Buildings**:
>    - The market is situated in front of a multi-story building with a light green facade. The building has several windows, some of which are open, allowing a glimpse into the interior.
> 
> 6. **Street**:
>    - The market is located on a brick-paved street. The street is clean and well-maintained, with a few yellow traffic cones visible near the entrance.
> 
> ### Scene Context:
> - The image was taken during the daytime, as indicated by the natural light and the clear visibility of the surroundings.
> - The market appears to be a popular destination, as evidenced by the large number of people present.
> 
> ### Relationships:
> - The neon signs and the clock are fixed structures that define the market's identity and serve as landmarks.
> - The people are interacting with the market, either by shopping, browsing, or simply passing through.
> - The market is integrated into the surrounding urban environment, with the building and street forming a cohesive setting.
> 
> ### Visual Attributes:
> - The colors are vivid, with the bright neon signs standing out against the more muted tones of the building and street.
> - The image captures the lively and dynamic nature of the market, with the movement of people and the variety of goods on display.
> 
> ### Uncertainty:
> - The image does not provide any specific information about the time of

### ln · idx 33079 · ln/oi_9ac675134bedf0a2 · L=512 · eos_hit=False

> This image captures a festive scene at night, featuring a large, elaborately decorated Christmas tree in the foreground and a prominent building in the background, which appears to be the United States Capitol building.
> 
> ### Main Objects:
> 1. **Christmas Tree**:
>    - The tree is tall and adorned with a multitude of colorful lights, including red, green, blue, and white, creating a vibrant and festive display.
>    - It is decorated with various ornaments, including what appear to be small lights, possibly small bulbs or baubles, and possibly some larger decorative items like stars or other shapes.
>    - At the very top of the tree, there is a large, illuminated star, which serves as the tree topper, adding to the festive glow.
> 
> 2. **United States Capitol Building**:
>    - The building is partially visible in the background, showcasing its iconic dome and columns.
>    - The dome is illuminated, with light streaming through the windows, giving it a warm and inviting appearance.
>    - The American flag is visible on a flagpole to the left of the dome, adding a patriotic element to the scene.
> 
> ### Scene Context:
> - The image is taken at night, as indicated by the dark sky and the artificial lighting from the tree and the building.
> - The setting appears to be a public space, likely in front of the Capitol building, given the proximity and the scale of the tree.
> - The overall atmosphere is celebratory and festive, likely during the holiday season.
> 
> ### Relationships:
> - The Christmas tree is positioned in the foreground, drawing the viewer's attention immediately.
> - The Capitol building is in the background, providing a sense of scale and context to the scene.
> - The tree is likely placed there as part of a holiday display, possibly organized by the Capitol or a local government body, given the grandeur and the official nature of the building.
> 
> ### Visual Attributes:
> - The colors are vivid, with the bright lights of the tree contrasting sharply against the dark night sky.
> - The Capitol building is well-lit, with the dome and columns clearly visible, emphasizing its architectural details.
> - The overall composition is balanced, with the tree on the right and the Capitol on the left, creating a symmetrical yet dynamic scene.
> 
> ### Uncertainty:
> - The exact location and the specific year the photo was taken are not discernible from the image alone.
> - The specific details of the ornaments on the tree, such as their exact shapes and sizes, are not clearly visible due to the distance and the lighting.
> 
> This image beautifully captures the
