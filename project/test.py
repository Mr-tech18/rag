lis=[{
    'id':"123",
    "content":"bonjour le monde"
},
    { 'id':"124",
    "content":"bonjour le monde1"
}]

r=[ra for r in lis for ra in r.values()]

print(r)