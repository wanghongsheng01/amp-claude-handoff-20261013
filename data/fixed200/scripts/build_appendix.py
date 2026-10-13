import fitz,json
from pathlib import Path
R=Path(__file__).resolve().parent;P=R/'package';doc=fitz.open();page=doc.new_page(width=595,height=842)
NAVY=(.09,.17,.28);TEAL=(.02,.42,.43);GRAY=(.33,.38,.42)
def box(text,x,y,w,h,size=10,color=NAVY):
 z=page.insert_textbox(fitz.Rect(x,y,x+w,y+h),text,fontname='helv' if text.isascii() else 'china-s',fontsize=size,lineheight=1.28,color=color)
 assert z>=0,(text[:40],z)
def table(y,widths,rows):
 x0=38;rh=22
 for i,row in enumerate(rows):
  x=x0
  for width,t in zip(widths,row):
   page.draw_rect(fitz.Rect(x,y+i*rh,x+width,y+(i+1)*rh),color=(.78,.82,.85),fill=(.91,.95,.96) if i==0 else (1,1,1),width=.35)
   box(str(t),x+5,y+i*rh+4,width-10,rh-4,9)
   x+=width
 return y+len(rows)*rh
page.draw_rect(fitz.Rect(0,0,595,12),fill=TEAL,color=TEAL)
box('AMP H5｜固定 200 数据准入附录',38,30,519,30,19)
box('2026-10-12 · 私有数据交付 · 无新采集/模型运行 · 科学结论 NO_GO 不变',38,66,519,22,10,GRAY)
box('一、实际增量（ID 不等于独立肽；测量行更不等于独立肽）',38,100,519,22,11,TEAL)
table(127,[195,80,80,80,84],[['项目','136 快照','146 快照','最终 200','较 146 增量'],['详情 / 肽 ID','136','146','200','+54'],['精确大小写序列','131','140','192','+52'],['序列家族 / 来源关联组','93 / 46','102 / 52','141 / 68','+39 / +16'],['全部 MIC 测量行','1242','1278','1788','+510'],['大肠杆菌 MIC 行','232','240','318','+78']])
box('较 136 快照：+64 个 ID、+546 行 MIC、+86 行大肠杆菌 MIC。',38,269,519,22,10)
box('二、原生单位资格计数（沿用冻结规则；未作质量—摩尔换算）',38,303,519,22,11,TEAL)
table(330,[195,80,80,80,84],[['单位 / 值型','测量行','肽 ID','家族','来源关联组'],['ug/ml · 点值','103','71','49','27'],['uM · 点值','88','42','28','19'],['ug/ml · 单侧界限','15','15','14','6'],['uM · 单侧界限','24','18','18','10']])
box('点值较 146：+24 / +18 行；较 136：+24 / +20 行（ug/ml / uM）。\n两单位组、点值与界限组可能重叠，不能直接相加。来源关联组并非已证实的生物独立样本。',38,451,519,43,10)
box('三、语义与缺失：保留而不推断',38,506,519,22,11,TEAL)
box('全部 MIC：1203 点值、336 右侧界限、215 范围、33 个 +/-、1 未解析；无左侧界限。\n大肠杆菌：226 点值、43 右侧界限、44 范围、5 个 +/-（上述为资格筛选前）。\n保留 > / >= 等原操作符；范围不取中点、不自动视作区间删失，+/- 不猜 SD/SE。\n两条空引用仍未解析：25752/204864（非大肠杆菌，8–16）；25758/204939\n（大肠杆菌，16–32）。唯一文献数组条目不足以补 reference=1，本地未找到原文。',38,533,519,88,10)
box('四、准入边界与交付内容',38,632,519,22,11,TEAL)
box('200/200 响应逐条通过哈希、ID、Monomer、JSON 校验，采集缺失为 0。\n仍有条件缺失：点值组培养基缺 3/14、pH 缺 103/88；端型缺失不等于 Free。\n固定位置样本不代表全库；这些是已暴露开发材料，未建立新拆分或完成污染隔离。\n新增数据不足以改变 NO_GO。原 27 条是否历史模型生成及筛选规则待作者回答，\n在回答前证据等级不变。未读 test37、未修改稿件、未重跑失败模型。\n小包含最小分析表、数据字典、离线作者清洗脚本、来源索引、哈希与独立核验。',38,658,519,105,10)
box('来源：DBAASP，2026-10-12 缓存；其数据政策允许使用/改编/再分发并要求署名。\ndbaasp.org/terms-and-conditions · Pirtskhalava et al. (2021), doi:10.1093/nar/gkaa991\n仅本对话私有 Library 附件；不含第三方原始 PDF。完整归因与限制见包内 README。',38,779,519,52,8,GRAY)
doc.save(P/'APPENDIX_1PAGE.pdf');page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(R/'appendix_preview.png');doc.close()
check=fitz.open(P/'APPENDIX_1PAGE.pdf');assert len(check)==1
out={'pages':len(check),'text_blocks':len(check[0].get_text('blocks')),'out_of_page_blocks':[b[:4] for b in check[0].get_text('blocks') if b[0]<0 or b[1]<0 or b[2]>595 or b[3]>842]};assert not out['out_of_page_blocks'];(R/'PDF_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
