import redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0,password='123456')


class RedisSave:
    def __init__(self):
        self.How_many_conversations_to_save=30

    def recent_messages(self):
        recent_messages =redis_client.lrange('recent_chat', 0, self.How_many_conversations_to_save - 1)
        return recent_messages

    def redis_store_chat(self,message):
        redis_client.lpush('recent_chat', message)
        redis_client.ltrim('recent_chat', 0, self.How_many_conversations_to_save - 1)

    def save_message_to_redis(self,message):
        message='Bot: '+message
        self.redis_store_chat(message)

    # def last_n_chat_messages(self,n):
    #     recent_message = redis_client.lindex('recent_chat', n)
    #     return recent_message if recent_message else None