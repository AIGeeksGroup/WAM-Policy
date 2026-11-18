"""Optional upload of student exports to a Hugging Face model repository; off unless configured.

Enable it with a ``hub`` section in the site file::

    "hub": {"repo": "<namespace>/<model-repo>", "path_in_repo": "arm_libero/<run-name>", "revision": "main"}

Rank 0 uploads each exported ``students/update-N[-ema]`` directory from one background thread, so
training never waits for the network; a failed upload is logged and never stops training. Credentials
come from the environment (``HF_TOKEN`` or a cached ``hf auth login``).
"""
import queue
import threading
import traceback


class SnapshotUploader:
    def __init__(self, repo, path_in_repo, revision=None, api=None):
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        self.api, self.repo, self.prefix, self.revision = api, repo, path_in_repo.strip("/"), revision
        self.jobs, self.events = queue.Queue(), queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @classmethod
    def from_site(cls, hub):
        """None unless the site file names a repository."""
        if not hub or not hub.get("repo"):
            return None
        return cls(hub["repo"], hub.get("path_in_repo", "arm_libero"), hub.get("revision"))

    def submit(self, update, snapshots):
        keys = []
        for variant, path in snapshots.items():
            key = f"{self.prefix}/{path.name}"
            self.jobs.put((update, variant, path, key))
            keys.append(key)
        return keys

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            update, variant, path, key = job
            try:
                self.api.upload_folder(repo_id=self.repo, folder_path=str(path), path_in_repo=key,
                                       revision=self.revision, commit_message=f"student update {update} ({variant})")
                self.events.put({"event": "hub_upload", "student_update": update, "variant": variant,
                                 "path_in_repo": key, "status": "uploaded"})
            except Exception as exc:
                traceback.print_exc()
                self.events.put({"event": "hub_upload", "student_update": update, "variant": variant,
                                 "path_in_repo": key, "status": "failed", "error": repr(exc)})

    def poll(self):
        records = []
        while not self.events.empty():
            records.append(self.events.get())
        return records

    def close(self, wait=True):
        self.jobs.put(None)
        if wait:
            self.thread.join()
        return self.jobs.qsize()
