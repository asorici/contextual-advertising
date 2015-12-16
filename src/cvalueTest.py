import ioData, codecs

df = ioData.readData("dataset/preProcData.json")
f = codecs.open("dataset/cValueTest_small.txt", "w", "utf-8")

print("Starting processing ...")

for index, content in enumerate(df.textZone[1:20]):
    for par in content:
        par_text = reduce(lambda x, y: x + y, par, "")
        f.write(par_text)
        f.write("\n")

    f.write("##########END##########")
    f.write("\n")

    print("Done processing page %i" % index)

print("Done processing!")
f.close()
