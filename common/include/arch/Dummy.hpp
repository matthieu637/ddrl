#ifndef DUMMY_H
#define DUMMY_H

namespace arch {

class DummyProgOptions {
public:
    static boost::program_options::options_description program_options() {
        boost::program_options::options_description desc;
        return desc;
    }
};

class AgentProgOptions {
public:
    static boost::program_options::options_description program_options() {
        boost::program_options::options_description desc("Allowed Agent options");
        desc.add_options()
        ("load", boost::program_options::value<std::string>(), "set the agent to load");
        return desc;
    }
};

}

#endif // DUMMY_H
