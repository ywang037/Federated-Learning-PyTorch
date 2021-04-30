import os
import keyboard  # using module keyboard

message_0 = 'you are ok to go'
message_1 = 'aborted'
os.system('pause')
print('press y to go, n to abort')
if keyboard.is_pressed('y'):  # if key 'q' is pressed 
    print(message_0)
elif keyboard.is_pressed('n'):
    exit(message_1)  # finishing the loop

print('if you pres y you will see program ends here')