import store
c = store.connect()
c.execute("DELETE FROM usage WHERE run_id = 'diag-1'")
c.commit()
c.close()
print("removed diag row")
