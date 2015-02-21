#ifndef COMMONAE_H
#define COMMONAE_H

class CommonAE
{
public :
    std::ostream& display(std::ostream& out, bool display, bool dump) {
        if(display) {
            _display(out);
        }

        if(dump) {
            _dump(out);
        }

        return out;
    }
protected:
    virtual void _display(std::ostream&){
	
    }
    
    virtual void _dump(std::ostream&){
	
    }
};

#endif // COMMONAE_H
