from module.dialogue_system import DialogueSystem

def main():
    # 初始化对话系统
    model_name = "qwen/Qwen2-1.5B-Instruct"
    dialogue_system = DialogueSystem(model_name, mode="normal")

    # 循环进行对话
    while True:
        user_input = input("用户: ")
        if user_input.lower() == "exit":
            print("对话结束。")
            break

        # 生成回复
        response = dialogue_system.generate_response(user_input)

        # 打印回复
        print("助手:", response)

if __name__ == "__main__":
    main()
