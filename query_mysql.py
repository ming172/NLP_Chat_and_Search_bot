from sqlalchemy import select
from create_json_to_mysql import Knowledge,session


class MySql:
    def __init__(self):
        pass

    def query_names(self,name_to_find):
        # 创建查询对象，使用 like 方法进行模糊匹配
        stmt = select(Knowledge.name).where(Knowledge.name.like(f"%{name_to_find}%"))
        # 执行查询
        result = session.execute(stmt)
        # 获取查询结果
        names = result.scalars().all()
        return names


    def name_list(self,names):
        # 打印查询结果
        list=[]
        list.append("符合条件的名称列表：")
        for idx, name in enumerate(names):
            list.append(f"{idx + 1}. {name}")
        return list
        # 让用户选择一个名称

    def choose_name(self,choice,names):
        # choice = input("请选择一个名称的序号：")
        if choice.isdigit() and int(choice) in range(1, len(names) + 1):
            return names[int(choice) - 1]
        else:
            return "输入无效，请重新选择"


    def query_description(self,name):
        # 创建查询对象
        stmt = select(Knowledge.description).where(Knowledge.name == name)

        # 执行查询
        result = session.execute(stmt)

        # 获取查询结果
        description = result.scalar()

        return description


