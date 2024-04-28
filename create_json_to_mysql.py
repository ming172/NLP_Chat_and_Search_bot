from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
import json
# 创建一个 SQLAlchemy 引擎并连接到 MySQL 数据库
engine = create_engine('mysql://root:123456@localhost:3307/datasql')#连接到名为datasql的数据库
# 创建基类
Base = declarative_base()#曾经的Base = declarative_base()
# 定义一个模型类，用来映射数据库表
class Knowledge(Base):
    __tablename__ = 'knowledge'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    property = Column(String(255))
    to_name = Column(String(255))
    importance = Column(Integer)
    description = Column(String(1000))

if __name__=='__main__':
    # 创建数据表
    Base.metadata.create_all(engine)

# 读取 JSON 文件并插入数据库
with open('computer_knowledge/JSONfromHTML.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

Session = sessionmaker(bind=engine)
session = Session()

for key, value in data.items():
    knowledge = Knowledge(
        name=key,
        property=value['property'][0] if 'property' in value else None,
        to_name=value['to_name'][0] if 'to_name' in value else None,
        importance=value['重要程度'][0] if '重要程度' in value else None,
        description=value['知识点描述'][0] if "知识点描述" in value and value["知识点描述"] else None,
    )
    if __name__ == '__main__':
        session.add(knowledge)

session.commit()
session.close()
if __name__=='__main__':
    print('完成，请刷新数据库'+' '+'Done, please refresh the database')
