# Run this to add SuperSR package into system path.
# You can import this package directly.
if __name__ == "__main__":
    import sys  
    import os
    if not os.getcwd() in sys.path:  
        sys.path.append(os.getcwd())   
    if not 'SuperSR' in sys.modules:
        a = __import__('SuperSR')  
    else:  
        eval('import SuperSR')  
        a = eval('reload(SuperSR)')  