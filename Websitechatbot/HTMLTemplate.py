# HTMLTemplate.py

css = '''
<style>
.chat-message {
    padding: 1rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex; 
    align-items: flex-start; /* Align items to the start */
}
.chat-message.user {
    background-color: #f0f0f0;
    color: #333;
}
.chat-message.bot {
    background-color: #e8e8e8;
}
.chat-message .avatar {
    flex-shrink: 0;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    margin-right: 10px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 24px;
}
.chat-message .message {
    flex-grow: 1;
    padding: 0.5rem 1rem;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">🤖</div>
    <div class="message"><strong>Bot:</strong><br>{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">👤</div>
    <div class="message"><strong>You:</strong><br>{{MSG}}</div>
</div>
'''
