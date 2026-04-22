"""
@Author  :  CAISIMIN
@Date    :  2026/4/19 21:26
"""
import sqlite3
from typing import Union, List
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text  # ORM框架
import pandas as pd
import time
import jwt
import requests
from itertools import combinations
import numpy as np
from tqdm import tqdm


conn = sqlite3.connect('chinook.db')

# 创建一个游标对象
cursor = conn.cursor()



# 数据库解析
print("=======================数据库解析=====================")
class DBParser:
    '''DBParser'''
    def __init__(self, db_url:str) -> None:
        '''初始化
        db_url: 数据库链接地址
        '''

        # 根据 URL 判断数据库类型
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'

        # 链接数据库
        self.engine = create_engine(db_url, echo=False) # 创建数据库引擎，echo=False 表示不打印 SQL 日志
        self.conn = self.engine.connect() # 建立数据库连接
        self.db_url = db_url # 保存数据库 URL

        self.inspector = inspect(self.engine)  # 创建数据库检查器对象
        self.table_names = self.inspector.get_table_names()  # 获取数据库中所有表的名称列表

        self._table_fields = {}  # 数据表字段, 存储每张表的字段信息
        self.foreign_keys = []  # 数据库外键，存储外键关系
        self._table_sample = {}  # 数据表样例，存储每张表的样例数据
        self._table_sample_count = {} # 数据表样例数量

        # 依次对每张表的字段进行统计
        for table_name in self.table_names:
            print("Table ->", table_name)
            self._table_fields[table_name] = {}

            self.foreign_keys += [  # 获取当前表的所有外键关系
                {
                    'constrained_table': table_name,  # 当前表名
                    'constrained_columns': x['constrained_columns'],  # 当前表的外键列
                    'referred_table': x['referred_table'],  # 被引用的表名
                    'referred_columns': x['referred_columns'],  # 被引用的列名
                } for x in self.inspector.get_foreign_keys(table_name)
            ]

            # 获取当前表的字段信息
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)  # 加载表的完整结构信息
            table_columns = self.inspector.get_columns(table_name)  # 获取所有列的元数据（名称、类型、是否可空等）
            self._table_fields[table_name] = {x['name']: x for x in table_columns}  # 将列信息转换为字典，以列名为键

            # 对当前字段进行统计
            for column_meta in table_columns:
                column_instance = getattr(table_instance.columns, column_meta['name']) # 获取字段的列对象，用于构建 SQL 查询
                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count

                field_type = self._table_fields[table_name][column_meta['name']][
                    'type']  # 从之前存储的字段信息字典中取出 type 属性，如：VARCHAR(50)、TEXT、INTEGER、DATETIME 等
                field_type = str(field_type)  # SQLAlchemy 的类型对象转为字符串
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    top1_value = self.conn.execute(query).fetchone()[0]
                    self._table_fields[table_name][column_meta['name']]['mode'] = top1_value

                # 统计missing个数
                query = select(func.count()).filter(column_instance == None)
                nan_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['nan_count'] = nan_count

                # 统计max
                query = select(func.max(column_instance))
                max_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['max'] = max_value

                # 统计min
                query = select(func.min(column_instance))
                min_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['min'] = min_value

                query = select(column_instance).limit(10)
                random_value = self.conn.execute(query).all()
                random_value = [x[0] for x in random_value]
                random_value = [str(x) for x in random_value if x is not None] # 转换为字符串并过滤空值；不同字段类型不同（整数、浮点数、日期等）； 统一转为字符串便于后续展示和使用
                random_value = list(set(random_value))
                self._table_fields[table_name][column_meta['name']]['random'] = random_value[:3]

            # 获取表样例（第一行）
            query = select(table_instance)  # SELECT * FROM table_nam
            self._table_sample[table_name] = pd.DataFrame([self.conn.execute(query).fetchone()]) # [...]：将元组包装成列表，pd.DataFrame() 需要接收二维数据结构
            self._table_sample[table_name].columns = [x['name'] for x in table_columns] # 设置正确的列名

            # 统计各表的样例数量
            query = select(func.count()).select_from(table_instance)
            self._table_sample_count[table_name] = self.conn.execute(query).fetchone()[0]

        # 获取数据库中表的数量
        self._tabele_count = len(self.table_names)

    def get_table_fields(self, table_name) -> pd.DataFrame:
        '''获取表字段信息'''
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        '''获取数据库链接信息（主键和外键）'''
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        '''获取数据表样例'''
        return self._table_sample[table_name]

    def get_tabel_names(self)-> List[str]:
        return self.table_names()

    def check_sql(self, sql) -> Union[bool, str]:  # 待办？？利用一些sql的工具进行检查
        '''检查sql是否合理

        参数
            sql: 待执行句子

        返回: 是否可以运行 报错信息
        '''
        try:
            self.conn.execute(text(sql))
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> bool:
        '''运行SQL'''
        result = self.conn.execute(text(sql))
        return list(result)

def ask_glm(question, nretry=5):
    if nretry == 0:
        return None

    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
      'Content-Type': 'application/json',
      'Authorization': generate_token("ee0775cde4f14adbbc2a35e18c1f2e44.yPLv1YuvvUQiNPDL", 1000)
    }
    data = {
        "model": "glm-3-turbo",
        "p": 0.5,
        "messages": [{"role": "user", "content": question}]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10) # 10 秒内服务器没有返回响应，客户端就会主动断开连接并抛出异常。目的是防止程序无限期等待
        return response.json()
    except:
        return ask_glm(question, nretry-1)

# =============新增：智能SQL回答Agent=================
print("====================智能SQL回答Agent======================")

class SQLAgent:
    """智能SQL问答助手，能够处理元数据查询、单表查询和多表联合查询"""

    def __init__(self, db_parser: DBParser):
        self.parser = db_parser
        # 构建完整的数据库schema信息
        self.full_schema = self._build_full_schema()

    def _build_full_schema(self) -> str:
        """构建完整的数据库schema描述"""
        schema_parts = []

        # 添加表名列表
        schema_parts.append(f"数据库中共有 {len(self.parser.table_names)} 张表")
        schema_parts.append(f"表名列表: {', '.join(self.parser.table_names)}")
        schema_parts.append("\n各表详细信息:")

        # 添加每张表的详细信息
        for table_name in self.parser.table_names:
            schema_parts.append(f"\n### 表名: {table_name}")

            # 字段信息
            fields_df = self.parser.get_table_fields(table_name)
            schema_parts.append("字段信息:")
            for idx, row in fields_df.iterrows():
                field_info = f"  - {row.name}: 类型={row.get('type', 'N/A')}"
                if pd.notna(row.get('distinct')):
                    field_info += f", 唯一值数={int(row['distinct'])}"
                if pd.notna(row.get('nan_count')):
                    field_info += f", 空值数={int(row['nan_count'])}"
                schema_parts.append(field_info)

            # 样例数据
            sample_df = self.parser.get_table_sample(table_name)
            if not sample_df.empty:
                schema_parts.append("样例数据 (前3行):")
                schema_parts.append(sample_df.head(3).to_string(index=False))

        # 添加外键关系
        if self.parser.foreign_keys:
            schema_parts.append("\n### 表之间的关系 (外键):")
            relations_df = self.parser.get_data_relations()
            for _, row in relations_df.iterrows():
                schema_parts.append(
                    f"  {row['constrained_table']}.{row['constrained_columns']} "
                    f"-> {row['referred_table']}.{row['referred_columns']}"
                )

        return "\n".join(schema_parts)

    def generate_sql(self, question: str) -> str:
        """根据自然语言问题生成SQL"""

        system_prompt = """你是一个专业的SQL专家，擅长将自然语言问题转换为SQL查询。

请根据以下数据库schema信息，将用户的问题转换为正确的SQL语句。

重要规则：
1. 只输出SQL语句，不要有任何解释或其他文字
2. SQL语句语法必须是有效的
3. 如果问题是关于表数量的，使用: SELECT COUNT(*) FROM sqlite_master WHERE type='table'
4. 如果问题是关于某表记录数的，使用: SELECT COUNT(*) FROM 表名
5. 如果需要统计多个表的记录数，使用子查询或UNION
6. 确保表名和字段名完全匹配schema中的名称
7. 不要使用Markdown代码块标记，直接输出SQL

数据库Schema信息：
{schema}
"""

        user_prompt = f"问题: {question}\n\n请生成对应的SQL:"

        full_prompt = system_prompt.format(schema=self.full_schema) + "\n" + user_prompt

        try:
            response = ask_glm(full_prompt)
            if response and 'choices' in response:
                sql = response['choices'][0]['message']['content']
                # 清理SQL
                sql = sql.strip().strip('`').strip()
                sql = sql.replace('sql\n', '').replace('SQL\n', '')
                sql = sql.strip()
                return sql
        except Exception as e:
            print(f"生成SQL失败: {e}")

        return None

    def execute_and_format(self, question: str, sql: str) -> str:
        """执行SQL并将结果格式化为自然语言回答"""

        # 验证SQL
        is_valid, error_msg = self.parser.check_sql(sql)
        if not is_valid:
            return f"SQL验证失败: {error_msg}"

        # 执行SQL
        try:
            result = self.parser.execute_sql(sql)

            # 格式化结果
            format_prompt = """请将SQL查询结果转换为自然语言回答。

原始问题: {question}
执行的SQL: {sql}
查询结果: {result}

请用简洁的中文回答问题，直接给出答案。"""

            format_input = format_prompt.format(
                question=question,
                sql=sql,
                result=str(result)
            )

            response = ask_glm(format_input)
            if response and 'choices' in response:
                return response['choices'][0]['message']['content']
            else:
                return f"查询结果: {result}"

        except Exception as e:
            return f"执行SQL失败: {str(e)}"

    def answer(self, question: str) -> str:
        """完整流程：生成SQL -> 执行 -> 返回自然语言答案"""
        print(f"\n问题: {question}")

        # 生成SQL
        sql = self.generate_sql(question)
        if not sql:
            return "无法生成SQL语句"

        print(f"生成的SQL: {sql}")

        # 执行并格式化
        answer = self.execute_and_format(question, sql)
        print(f"答案: {answer}")

        return answer


# 模拟生成
print("=======================模拟生成=====================")
parser = DBParser('sqlite:///./chinook.db')

# 实际KEY，过期时间
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    # 使用 JWT 库将 payload 编码为加密的 Token 字符串
    return jwt.encode(
        payload, # Token 携带的数据
        secret, # 签名密钥，用于加密和验证
        algorithm="HS256", # HMAC-SHA256 算法，对称加密
        headers={"alg": "HS256", "sign_type": "SIGN"}, # 自定义头部，指定算法和签名类型
    )

# 创建SQL Agent实例
agent = SQLAgent(parser)

# 测试三个问题
questions = [
    "数据库中总共有多少张表？",
    "员工表中有多少条记录？",
    "在数据库中所有客户个数和员工个数分别是多少？"
]

print("\n" + "="*60)
print("开始测试智能SQL问答Agent")
print("="*60)

for q in questions:
    try:
        answer = agent.answer(q)
        print("-" * 60)
    except Exception as e:
        print(f"处理问题时出错: {e}")
        traceback.print_exc()
        print("-" * 60)

# def ask_glm(question, nretry=5):
#     if nretry == 0:
#         return None
#
#     url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
#     headers = {
#       'Content-Type': 'application/json',
#       'Authorization': generate_token("ee0775cde4f14adbbc2a35e18c1f2e44.yPLv1YuvvUQiNPDL", 1000)
#     }
#     data = {
#         "model": "glm-3-turbo",
#         "p": 0.5,
#         "messages": [{"role": "user", "content": question}]
#     }
#     try:
#         response = requests.post(url, headers=headers, json=data, timeout=10) # 10 秒内服务器没有返回响应，客户端就会主动断开连接并抛出异常。目的是防止程序无限期等待
#         return response.json()
#     except:
#         return ask_glm(question, nretry-1)
#
#
# database_prompt = '''你是一个专业的数据库专家，现在需要你结合数据库的信息和提问，生成对应的SQL语句。请直接输出SQL，不需要有其他输出：
#
# 数据库表名称如下：{table_names}
#
# 提问：{question}
# '''
#
# table_prompt = '''你是一个专业的数据库专家，现在需要你结合表{table_name}的信息和提问，生成对应的SQL语句。请直接输出SQL，不需要有其他输出：
#
# 表名称：{table_name}
#
# 数据表样例如下：
# {data_sample_mk}
#
# 数据表schema如下：
# {data_schema}
#
# 提问：{question}
# '''
#
# def generate_sql(prompt):
#     for attempt in range(5):
#         try:
#             print(f"  尝试第 {attempt + 1} 次...")
#             input_str = prompt
#             # 生成sql
#             response = ask_glm(input_str)
#             if response is None:
#                 print(f"  API返回None")
#                 continue
#
#             answer = response['choices'][0]['message']['content']
#             answer = answer.strip('`').strip('\n').replace('sql\n', '')
#             print(f"  生成的SQL: {answer}")
#
#             # 判断SQL是否符合逻辑
#             flag, error_msg = parser.check_sql(answer)
#             if not flag:
#                 print(f"  SQL验证失败: {error_msg}")
#                 continue
#
#             # 获取SQL答案
#             sql_answer = parser.execute_sql(answer)
#             print(f"  查询结果: {sql_answer}")
#             return sql_answer
#         except Exception as e:
#             print(f"  异常: {str(e)}")
#             continue
#
#     print("所有尝试均失败")
#     return None
#
# print("\n开始回答问题1:")
# question1 = "数据库中总共有多少张表？"
# input_str = database_prompt.format(table_names=parser.table_names, question=question1)
# result1 = generate_sql(input_str)
# print(f"问题1答案: {result1}\n")
#
# print("开始回答问题2:")
# question2 = "员工表中有多少条记录？"
# data_sample = parser.get_table_sample('employees')
# data_schema = parser.get_table_fields('employees')
# input_str2 = table_prompt.format(table_name='employees', data_sample_mk=data_sample.to_markdown(), data_schema=data_schema.to_markdown(), question=question2)
# result2 = generate_sql(input_str2)
# print(f"问题2答案: {result2}\n")
#
# print("开始回答问题3:")
# question3 = "在数据库中所有客户个数和员工个数分别是多少？"
# input_str3 = database_prompt.format(table_names=parser.table_names, question=question3)
# result3 = generate_sql(input_str3)
# print(f"问题3答案: {result3}\n")


