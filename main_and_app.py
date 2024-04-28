# app.py
from flask import Flask, request, jsonify, render_template,redirect, url_for
from config import Config
import output_answer as oa
import query_mysql
import redis_data as rd
import ChatLM

query=query_mysql.MySql()
redis_save=rd.RedisSave()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index5.html')


search=False;begin_search=False;names_save=''
choose_model=False;use_bigmodel=False
@app.route('/chat', methods=['POST'])
def chat(**kwargs):
    global search
    global begin_search
    global names_save
    global choose_model
    global use_bigmodel
    opt = Config()
    for k, v in kwargs.items():  # 设置参数
        setattr(opt, k, v)

    searcher, sos, eos, unknown, word2ix, ix2word = oa.test(opt)

    try:
        input_sentence = request.form['message']
        if input_sentence.lower() in ['q', 'quit', 'exit']:
            message='Bye! Have a good day!'
            redis_save.save_message_to_redis(message)
            return jsonify({'response': message})


        if input_sentence=='choose':
            message = '请选择序号：<br>1.小模型chatbot<br>2.中模型ChatLM-mini-Chinese'
            choose_model=True
            return jsonify({'response': message})
        if  choose_model==True:
            if input_sentence=='2':
                use_bigmodel=True
                choose_model=False
                message = '中模型ChatLM-mini-Chinese已开启'
                return jsonify({'response': message})
            elif input_sentence=='1':
                use_bigmodel = False
                choose_model = False
                message = '小模型chatbot已开启'
                return jsonify({'response': message})
            else:
                message = '输入错误，请重新输入(输入不带引号的“1”或“2”)'
                return jsonify({'response': message})

        if opt.mysql_QA:

            if input_sentence=='、、' and search==False:
                search=True
                message='进入查询模式，开始查询，请告诉我您要查询的问题(只输入关键字)'
                redis_save.save_message_to_redis(message)
                return jsonify({'response': message})
            elif search == True and begin_search==False:
                if input_sentence == '、、':
                    search = False
                    begin_search = False
                    message = '已退出查询模式'
                    redis_save.save_message_to_redis(message)
                    return jsonify({'response': message})

                names=query.query_names(input_sentence)
                names_save=names
                list=query.name_list(names)

                if list==["符合条件的名称列表："]:
                    message = '未找到您要查询的问题'
                    redis_save.save_message_to_redis(message)
                    return jsonify({'response': message})
                else:
                    a=''
                    for value in list:
                        a+=value+'<br>'
                    a+="请选择一个名称的序号："
                    begin_search = True
                    redis_save.save_message_to_redis(a)
                    return jsonify({'response': a })
            elif search == True and begin_search==True:
                if input_sentence=='、':
                    begin_search = False
                    message = "已成功返回上级，重新进入查询模式，请告诉我您要查询的问题(只输入关键字)"
                    redis_save.save_message_to_redis(message)
                    return jsonify({'response': message})
                names=names_save
                selected_name=query.choose_name(input_sentence, names)
                while selected_name=="输入无效，请重新选择":
                    if input_sentence=='、、':
                        search = False;begin_search=False
                        message = '已退出查询模式'
                        redis_save.save_message_to_redis(message)
                        return jsonify({'response': message})
                    else:
                        message = "输入无效，请重新选择"
                        redis_save.save_message_to_redis(message)
                        return jsonify({'response': message})#会继续进入这一判断语句elif search == True and begin_search==True:

                description = query.query_description(selected_name)
                redis_save.save_message_to_redis(description)
                return jsonify({'response': description})
            if input_sentence=='、、' and search == True:
                search = False
                message = '已退出查询模式'
                redis_save.save_message_to_redis(message)
                return jsonify({'response': message})
            else:
                if use_bigmodel==True:
                    output_words=ChatLM.use_ChatLM(input_sentence)
                else:
                    output_words = oa.output_answer(input_sentence, searcher, sos, eos, unknown, opt, word2ix, ix2word)
                redis_save.save_message_to_redis(output_words)
                return jsonify({'response': output_words})
        else:
            if use_bigmodel == True:
                output_words = ChatLM.use_ChatLM(input_sentence)
            else:
                output_words = oa.output_answer(input_sentence, searcher, sos, eos, unknown, opt, word2ix, ix2word)
            redis_save.save_message_to_redis(output_words)
            return jsonify({'response': output_words})
    except Exception as e:
        print(e)
        message = 'Sorry, something went wrong!'
        redis_save.save_message_to_redis(message)
        return jsonify({'response': message})



# 在路由中处理对话信息
@app.route('/store_chat', methods=['POST'])
def store_chat():
    message = 'You: '+request.form['message']
    # print("Received message:", message)
    redis_save.redis_store_chat(message)
    return redirect(url_for('home'))

# 查看最近n次对话信息
@app.route('/recent_chat')
def recent_chat():
    # 从Redis列表中获取最近n次的对话信息
    recent_messages =redis_save.recent_messages()
    messages_as_strings=[]
    for message in recent_messages:
        a=message.decode('utf-8')
        if a.startswith('Bot: 符'):
            a = a.replace('<br>', ' ')
        messages_as_strings.append(a)
    return render_template('recent_chat.html', messages=messages_as_strings)


if __name__ == "__main__":
    app.run(debug=True)
