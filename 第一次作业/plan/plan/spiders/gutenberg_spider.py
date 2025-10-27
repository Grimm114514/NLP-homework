# plan/spiders/gutenberg_spider.py

import scrapy
from plan.items import BookItem  # 导入你在 items.py 中定义的 Item

class GutenbergSpider(scrapy.Spider):
    name = 'gutenberg'
    allowed_domains = ['gutenberg.org']
    
    # 使用新的 start() 方法 (async def)
    async def start(self):
        """
        爬虫的入口点。
        """
        yield scrapy.Request(url='https://www.gutenberg.org/browse/scores/top', callback=self.parse_top100)

    def parse_top100(self, response):
        """
        这个函数解析 "Top 100" 列表页面。
        它会找到每本书的链接 (例如 /ebooks/1234)
        """
        # 【已修正】使用更准确的 CSS 选择器，只选择 /ebooks/ 开头的链接
        book_links = response.css('ol li a[href^="/ebooks/"]::attr(href)').getall()
        
        if not book_links:
            self.logger.error(f"在 {response.url} 上没有找到任何 book_links！CSS 选择器可能已失效。")
            return # 如果没有找到链接，就停止

        for link in book_links:
            # response.follow 会自动拼接成完整的 URL
            yield response.follow(link, callback=self.parse_book_page)

    def parse_book_page(self, response):
        """
        这个函数解析单个书本的详情页。
        它的目标是找到 "Plain Text UTF-8" 版本的链接。
        """
        # 提取书名
        title = response.css('h1[itemprop="name"]::text').get()
        if not title:
            title = response.css('h1::text').get() # 备用方案
            
        # 找到 "Plain Text UTF-8" 文件的链接
        text_link = response.xpath('//a[contains(text(), "Plain Text UTF-8")]/@href').get()

        if text_link:
            # 使用 meta 参数把书名和来源URL传递给下一个解析函数
            yield response.follow(
                text_link, 
                callback=self.parse_text, 
                meta={
                    'title': title, 
                    'source_url': response.url  # 传递详情页的URL，而不是txt文件的URL
                }
            )
        else:
            self.logger.warning(f"Could not find 'Plain Text UTF-8' link for: {response.url}")

    def parse_text(self, response):
        """
        这是最后一步，解析 .txt 文件。
        这个函数的核心是清理掉古腾堡的法律声明文本。
        """
        # 从 meta 中取回传递过来的数据
        title = response.meta.get('title', 'Unknown Title')
        source_url = response.meta.get('source_url', 'Unknown URL')
        
        full_text = response.text

        # --- 关键的清理步骤 ---
        start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
        
        if start_marker not in full_text:
            start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK" # 备用标记

        if end_marker not in full_text:
            end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK" # 备用标记

        start_index = full_text.find(start_marker)
        end_index = full_text.find(end_marker)

        clean_text = ""

        if start_index != -1 and end_index != -1:
            start_index += len(start_marker) # 移动到标记之后
            clean_text = full_text[start_index:end_index].strip()
        else:
            self.logger.warning(f"Could not find start/end markers in: {response.url}. Using raw text.")
            clean_text = full_text # 回退方案
        
        # --- 清理完毕 ---

        item = BookItem()
        item['title'] = title.strip() if title else 'Unknown Title'
        item['url'] = source_url # 保存书本详情页的URL
        item['text_content'] = clean_text
        
        yield item