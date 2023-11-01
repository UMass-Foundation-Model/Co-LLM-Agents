from pathlib import Path
from py_md_doc import PyMdDoc
from md_link_tester import MdLinkTester

if __name__ == "__main__":
    # API documentation.
    md = PyMdDoc(input_directory=Path("transport_challenge_multi_agent"),
                 files=[Path("transport_challenge_multi_agent/challenge_state.py").resolve(),
                        Path("transport_challenge_multi_agent/multi_action.py").resolve(),
                        Path("transport_challenge_multi_agent/navigate_to.py").resolve(),
                        Path("transport_challenge_multi_agent/pick_up.py").resolve(),
                        Path("transport_challenge_multi_agent/put_in.py").resolve(),
                        Path("transport_challenge_multi_agent/reach_for_transport_challenge.py").resolve(),
                        Path("transport_challenge_multi_agent/replicant_target_position.py").resolve(),
                        Path("transport_challenge_multi_agent/replicant_transport_challenge.py").resolve(),
                        Path("transport_challenge_multi_agent/reset_arms.py").resolve(),
                        Path("transport_challenge_multi_agent/transport_challenge.py").resolve()])
    md.get_docs(output_directory=Path(f"doc"), import_prefix="from transport_challenge_multi_agent")
    # Test links.
    links = MdLinkTester.test_file(Path("README.md"))
    if len(links) > 0:
        print("README.md:")
        for link in links:
            print("\t" + link)
    for f in Path("doc").iterdir():
        if f.suffix == ".md":
            links = MdLinkTester.test_file(f)
            if len(links) > 0:
                print(f)
                for link in links:
                    print("\t" + link)
